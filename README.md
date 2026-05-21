Markdown# ModeloHaptic

Sistema de detección de objetos en tiempo real utilizando YOLO, OpenCV y Arduino, optimizado mediante el filtrado dinámico y tunneo (*fine-tuning*) del dataset COCO para asistencia háptica.

---

## Descripción técnica

**ModeloHaptic** es un sistema de visión por computadora diseñado para la interacción física y la retroalimentación háptica en entornos urbanos. 

A diferencia de un uso genérico de YOLO con sus 80 clases predeterminadas de COCO, este proyecto implementa un **proceso de tunneo y filtrado selectivo del dataset**. Se extrajeron y reentrenaron únicamente las clases críticas para la navegación espacial y la movilidad, optimizando el rendimiento del modelo para dispositivos de procesamiento en tiempo real.

### Clases seleccionadas y optimizadas:
* `person` (Persona)
* `bicycle` (Bicicleta)
* `car` (Carro)
* `motorcycle` (Motocicleta)
* `bus` (Autobús)
* `train` (Tren)
* `truck` (Camión)
* `traffic light` (Semáforo)
* `fire hydrant` (Hidrante)
* `stop sign` (Señal de alto)
* `bench` (Banca)

---

## ¿Por qué tunear el dataset COCO?

El dataset COCO original contiene decenas de categorías que no aportan valor a un sistema de navegación (como "sofá" o "manzana"). El proceso de filtrado y ajuste fino (*fine-tuning*) se realizó para:
1.  **Reducir la carga computacional:** Menos clases implican una inferencia más ligera en entornos embebidos.
2.  **Optimizar la velocidad (FPS):** Crucial para que la respuesta háptica de los motores en el Arduino sea inmediata.
3.  **Minimizar falsos positivos:** Evita que objetos irrelevantes del entorno confundan al usuario.

---

## Arquitectura del sistema

```text
Dataset COCO Original (80 Clases)
               ↓
 Filtrado de Clases de Movilidad (11 Clases)
               ↓
   Fine-Tuning del Modelo YOLO
               ↓
     Modelo Optimizado (best.pt)
               ↓
  Captura de Cámara en Tiempo Real
               ↓
   Extracción del Centro Geométrico
               ↓
     Estructuración de Datos (JSON)
               ↓
   Comunicación Serial UART (PySerial)
               ↓
 Arduino / Activación de Motores Hápticos
Análisis y Distribución del DatasetPara validar el comportamiento geométrico y la consistencia de las anotaciones antes y durante el entrenamiento, se analizó la distribución espacial de las cajas delimitadoras (bounding boxes).Estadísticas de las EtiquetasLa siguiente sección muestra la cantidad de instancias por clase, las dimensiones relativas de las cajas y su dispersión o concentración en el plano de los fotogramas.Muestras del Set de EntrenamientoA continuación se observa un mosaico de imágenes procesadas por el pipeline de Ultralytics, verificando el correcto etiquetado y el proceso de data augmentation aplicado a las clases urbanas seleccionadas.Evaluación del ModeloEl rendimiento del modelo final fue evaluado cuantitativamente utilizando métricas de precisión y exhaustividad.Curva F1-ConfidenceLa curva F1 mide el balance óptimo entre la Precisión (Precision) y la Recuperación (Recall). Tras el proceso de tunneo, el modelo alcanzó su punto de equilibrio óptimo bajo los siguientes parámetros:F1-Score Global: 0.63Umbral de Confianza (Confidence Threshold): 0.242Nota de rendimiento: Las clases con estructuras visuales rígidas y distintivas como train, fire hydrant y stop sign presentan el mejor desempeño. Clases con alta variabilidad morfológica o escala (como bench y traffic light) muestran un margen de mejora que se abordará en futuras iteraciones.Extracción Espacial de CoordenadasEl script en Python procesa el frame, detecta el objeto mediante el modelo tuneado best.pt y calcula su centro geométrico para determinar la ubicación del estímulo háptico:Python# Obtención de coordenadas de la caja delimitadora (Bounding Box)
x1, y1, x2, y2 = box.xyxy[0].tolist()

# Cálculo del centro geométrico del objeto
centro_x = x1 + (x2 - x1) / 2
centro_y = y1 + (y2 - y1) / 2
Estructura de Datos Transmitida (JSON)La posición del objeto se empaqueta en formato JSON para facilitar su lectura en el microcontrolador:JSON[
  {
    "objeto": "car",
    "centroObj": [420.3, 210.8]
  }
]
Comunicación SerialLos datos estructurados se envían por ráfagas a través del puerto UART:Pythonarduino.write((datos_json + "\n").encode('utf-8'))
ParámetroValorBaud Rate9600ComunicaciónUART SerialEncodingUTF-8Tecnologías UtilizadasUltralytics YOLO: Framework para el fine-tuning e inferencia del modelo de visión.OpenCV: Captura de video de la cámara web y renderizado de las anotaciones en tiempo real.PySerial: Protocolo de comunicación para transferir las coordenadas de los objetos al hardware.Arduino: Procesamiento físico de los datos para la activación de actuadores/motores vibratorios.JSON: Formato ligero de intercambio de datos entre Python y Arduino.Estructura del ProyectoEl repositorio está organizado de la siguiente manera:Plaintextmodelohaptic/
│
├── images/
│   ├── dataset_samples.png
│   ├── dataset_stats.png
│   └── F1_curve.png
│
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt
│
├── YoloCustom.py
├── YoloAPF.py
├── requirements.txt
└── README.md
Instalación y Uso1. Clonar el repositorioBashgit clone [https://github.com/Pollitobello2005/modelohaptic.git](https://github.com/Pollitobello2005/modelohaptic.git)
cd modelohaptic
2. Instalar dependencias requeridasBashpip install -r requirements.txt
3. Configuración del Puerto SerialAbre el archivo YoloCustom.py y asegúrate de que el puerto apunte al COM correspondiente de tu Arduino:PythonPUERTO_SERIAL = "COM3"  # Cambiar según corresponda (ej. /dev/ttyUSB0 en Linux)
BAUD_RATE = 9600
4. Ejecución del sistemaBashpython YoloCustom.py
Para cerrar la ventana de visualización en vivo, presiona la tecla Q.Aplicaciones PotencialesDispositivos de asistencia a la navegación para personas con discapacidad visual.Guiado táctil en robótica autónoma y vehículos de escala reducida.Interfaces avanzadas de interacción Humano-Máquina (HMI).
