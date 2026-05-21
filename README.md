# ModeloHaptic

Sistema de detección de objetos en tiempo real utilizando YOLO, OpenCV y Arduino, optimizado mediante filtrado dinámico y *fine-tuning* del dataset COCO para asistencia háptica.

---

## Descripción técnica

**ModeloHaptic** es un sistema de visión por computadora diseñado para la interacción física y la retroalimentación háptica en entornos urbanos.

A diferencia de un uso genérico de YOLO con sus 80 clases predeterminadas de COCO, este proyecto implementa un **proceso de fine-tuning y filtrado selectivo del dataset**. Se extrajeron y reentrenaron únicamente las clases críticas para la navegación espacial y la movilidad, optimizando el rendimiento del modelo para dispositivos de procesamiento en tiempo real.

### Clases seleccionadas y optimizadas

| Clase           | Descripción       |
|-----------------|-------------------|
| `person`        | Persona           |
| `bicycle`       | Bicicleta         |
| `car`           | Carro             |
| `motorcycle`    | Motocicleta       |
| `bus`           | Autobús           |
| `train`         | Tren              |
| `truck`         | Camión            |
| `traffic light` | Semáforo          |
| `fire hydrant`  | Hidrante          |
| `stop sign`     | Señal de alto     |
| `bench`         | Banca             |

---

## ¿Por qué tunear el dataset COCO?

El dataset COCO original contiene decenas de categorías que no aportan valor a un sistema de navegación (como "sofá" o "manzana"). El proceso de filtrado y ajuste fino se realizó para:

1. **Reducir la carga computacional:** Menos clases implican una inferencia más ligera en entornos embebidos.
2. **Optimizar la velocidad (FPS):** Crucial para que la respuesta háptica de los motores en el Arduino sea inmediata.
3. **Minimizar falsos positivos:** Evita que objetos irrelevantes del entorno confundan al usuario.

---

## Arquitectura del sistema

```
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
```

---

## Análisis y distribución del dataset

Para validar el comportamiento geométrico y la consistencia de las anotaciones antes y durante el entrenamiento, se analizó la distribución espacial de las cajas delimitadoras (*bounding boxes*).

### Estadísticas de las etiquetas

La siguiente sección muestra la cantidad de instancias por clase, las dimensiones relativas de las cajas y su dispersión o concentración en el plano de los fotogramas.

![Dataset Stats](runs/detect/train/labels.jpg)

### Muestras del set de entrenamiento

A continuación se observa un mosaico de imágenes procesadas por el pipeline de Ultralytics, verificando el correcto etiquetado y el proceso de *data augmentation* aplicado a las clases urbanas seleccionadas.

![Train Batch 0](runs/detect/train/train_batch0.jpg)
![Train Batch 1](runs/detect/train/train_batch1.jpg)
![Train Batch 2](runs/detect/train/train_batch2.jpg)

---

## Evaluación del modelo

El rendimiento del modelo final fue evaluado cuantitativamente utilizando métricas de precisión y exhaustividad.

### Curva F1-Confidence

La curva F1 mide el balance óptimo entre Precisión (*Precision*) y Recuperación (*Recall*). Tras el proceso de fine-tuning, el modelo alcanzó su punto de equilibrio óptimo bajo los siguientes parámetros:

- **F1-Score Global:** `0.63`
- **Umbral de Confianza:** `0.242`

![F1 Curve](runs/detect/train/BoxF1_curve.png)

> **Nota de rendimiento:** Las clases con estructuras visuales rígidas y distintivas como `train`, `fire hydrant` y `stop sign` presentan el mejor desempeño. Clases con alta variabilidad morfológica o escala (como `bench` y `traffic light`) muestran un margen de mejora que se abordará en futuras iteraciones.

### Curvas Precision / Recall

![Precision Curve](runs/detect/train/BoxP_curve.png)
![Recall Curve](runs/detect/train/BoxR_curve.png)
![PR Curve](runs/detect/train/BoxPR_curve.png)

### Resultados generales del entrenamiento

![Results](runs/detect/train/results.png)

### Matriz de confusión

![Confusion Matrix](runs/detect/train/confusion_matrix.png)
![Confusion Matrix Normalized](runs/detect/train/confusion_matrix_normalized.png)

### Predicciones de validación

| Labels | Predicciones |
|--------|--------------|
| ![val labels 0](runs/detect/train/val_batch0_labels.jpg) | ![val pred 0](runs/detect/train/val_batch0_pred.jpg) |
| ![val labels 1](runs/detect/train/val_batch1_labels.jpg) | ![val pred 1](runs/detect/train/val_batch1_pred.jpg) |
| ![val labels 2](runs/detect/train/val_batch2_labels.jpg) | ![val pred 2](runs/detect/train/val_batch2_pred.jpg) |

---

## Extracción espacial de coordenadas

El script en Python procesa el frame, detecta el objeto mediante el modelo tuneado `best.pt` y calcula su centro geométrico para determinar la ubicación del estímulo háptico:

```python
# Obtención de coordenadas de la caja delimitadora (Bounding Box)
x1, y1, x2, y2 = box.xyxy[0].tolist()

# Cálculo del centro geométrico del objeto
centro_x = x1 + (x2 - x1) / 2
centro_y = y1 + (y2 - y1) / 2
```

---

## Estructura de datos transmitida (JSON)

La posición del objeto se empaqueta en formato JSON para facilitar su lectura en el microcontrolador:

```json
[
  {
    "objeto": "car",
    "centroObj": [420.3, 210.8]
  }
]
```

---

## Comunicación serial

Los datos estructurados se envían por ráfagas a través del puerto UART:

```python
arduino.write((datos_json + "\n").encode('utf-8'))
```

| Parámetro     | Valor         |
|---------------|---------------|
| Baud Rate     | `9600`        |
| Comunicación  | UART Serial   |
| Encoding      | UTF-8         |

---

## Tecnologías utilizadas

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics):** Framework para el fine-tuning e inferencia del modelo de visión.
- **[OpenCV](https://opencv.org/):** Captura de video de la cámara web y renderizado de las anotaciones en tiempo real.
- **[PySerial](https://pyserial.readthedocs.io/):** Protocolo de comunicación para transferir las coordenadas de los objetos al hardware.
- **[Arduino](https://www.arduino.cc/):** Procesamiento físico de los datos para la activación de actuadores/motores vibratorios.
- **JSON:** Formato ligero de intercambio de datos entre Python y Arduino.

---

## Estructura del proyecto

```
modelohaptic/
│
├── runs/
│   └── detect/
│       └── train/
│           ├── weights/
│           │   └── best.pt
│           ├── BoxF1_curve.png
│           ├── BoxPR_curve.png
│           ├── BoxP_curve.png
│           ├── BoxR_curve.png
│           ├── confusion_matrix.png
│           ├── confusion_matrix_normalized.png
│           ├── labels.jpg
│           ├── results.png
│           ├── results.csv
│           ├── predictions.json
│           ├── args.yaml
│           ├── train_batch0.jpg
│           ├── train_batch1.jpg
│           ├── train_batch2.jpg
│           ├── val_batch0_labels.jpg
│           ├── val_batch0_pred.jpg
│           ├── val_batch1_labels.jpg
│           ├── val_batch1_pred.jpg
│           ├── val_batch2_labels.jpg
│           └── val_batch2_pred.jpg
│
├── .gitignore
├── YoloCustom.py
├── YoloAPF.py
├── requirements.txt
└── README.md
```

---

## Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/Pollitobello2005/modelohaptic.git
cd modelohaptic
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar el puerto serial

Abre el archivo `YoloCustom.py` y asegúrate de que el puerto apunte al COM correspondiente de tu Arduino:

```python
PUERTO_SERIAL = "COM3"  # Cambiar según corresponda (ej. /dev/ttyUSB0 en Linux)
BAUD_RATE = 9600
```

### 4. Ejecutar el sistema

```bash
python YoloCustom.py
```

> Para cerrar la ventana de visualización en vivo, presiona la tecla `Q`.

---

## Aplicaciones potenciales

- Dispositivos de asistencia a la navegación para personas con discapacidad visual.
- Guiado táctil en robótica autónoma y vehículos de escala reducida.
- Interfaces avanzadas de interacción Humano-Máquina (HMI).
