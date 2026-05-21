# ModeloHaptic

Sistema de detección de objetos en tiempo real utilizando YOLO, OpenCV y Arduino con optimización de clases mediante filtrado dinámico del dataset COCO.

---

# Descripción técnica

ModeloHaptic es un sistema de visión por computadora orientado a interacción física y retroalimentación háptica.

El proyecto utiliza un modelo YOLO basado en el dataset preentrenado COCO, el cual originalmente contiene aproximadamente 80 clases de detección. Sin embargo, el sistema implementa un proceso de *tunneo* y filtrado de clases para reducir únicamente a los objetos relevantes para el proyecto.

En lugar de entrenar un modelo completamente desde cero, se aprovecha el conocimiento previo aprendido por YOLO sobre COCO y posteriormente se realiza una especialización enfocada únicamente en ciertas categorías específicas.

Las clases utilizadas actualmente son:

- person
- bicycle
- car
- motorcycle
- bus
- train
- truck
- traffic light
- fire hydrant
- stop sign
- bench

---

# ¿Qué es el tunneo del modelo?

El proyecto implementa un proceso de optimización sobre el modelo YOLO preentrenado.

El dataset COCO contiene decenas de categorías generales, pero muchas de ellas no son útiles para el sistema háptico. Por esta razón, se realiza un filtrado de clases para:

- Reducir carga computacional
- Mejorar velocidad de inferencia
- Aumentar precisión sobre objetos relevantes
- Disminuir falsos positivos
- Optimizar el uso en tiempo real

El modelo conserva únicamente las clases necesarias relacionadas con movilidad, entorno urbano y navegación espacial.

---

# Arquitectura del sistema

```text
Dataset COCO
      ↓
Filtrado de clases relevantes
      ↓
Fine-Tuning / Tunneo del modelo YOLO
      ↓
Modelo optimizado (best.pt)
      ↓
Captura de cámara en tiempo real
      ↓
Detección de objetos
      ↓
Extracción espacial
      ↓
Comunicación serial
      ↓
Arduino / Sistema háptico
```

---

# Flujo interno del sistema

## 1. Modelo base COCO

YOLO inicia utilizando pesos preentrenados sobre COCO.

COCO proporciona conocimiento general sobre:

- personas
- vehículos
- señales
- objetos urbanos
- infraestructura vial

---

## 2. Filtrado de clases

El sistema elimina todas las categorías irrelevantes y conserva únicamente las necesarias para el proyecto.

Ejemplo conceptual:

```python
clases_utilizadas = [
    "person",
    "car",
    "bus",
    "motorcycle",
    "traffic light",
    "stop sign"
]
```

Esto permite especializar el detector únicamente sobre escenarios urbanos y navegación.

---

# Fine-Tuning del modelo

Después del filtrado se realiza un ajuste fino (*fine-tuning*).

El objetivo es que el modelo:

- mejore precisión sobre clases seleccionadas
- reduzca errores
- aprenda patrones específicos del entorno
- optimice inferencia en tiempo real

El resultado final se guarda en:

```text
runs/detect/train/weights/best.pt
```

---

# Evaluación del modelo

El sistema utiliza métricas F1-Score para evaluar el rendimiento de cada clase detectada.

## Curva F1-Confidence

La curva F1 mide el balance entre:

- Precision
- Recall

El mejor punto del modelo ocurre aproximadamente en:

```text
F1 = 0.63
Confidence Threshold = 0.242
```

Esto significa que el modelo obtiene el mejor equilibrio entre precisión y detección usando un umbral de confianza cercano a `0.24`.

Las clases con mejor desempeño son:

- train
- fire hydrant
- stop sign
- bus

Mientras que clases más complejas como:

- bench
- bicycle
- traffic light

presentan menor desempeño debido a variaciones visuales y tamaño en imagen.

---

# Funcionamiento del detector

## Captura de video

```python
cap = cv2.VideoCapture(0)
```

La cámara obtiene frames continuamente.

---

## Inferencia con YOLO

```python
results = model(frame)
```

Cada frame es procesado por el modelo optimizado.

---

## Bounding Boxes

YOLO devuelve:

- coordenadas
- clase
- confianza
- dimensiones del objeto

---

# Extracción espacial

El sistema calcula el centro geométrico del objeto detectado.

```python
centro_x = x1 + (x2 - x1) / 2
centro_y = y1 + (y2 - y1) / 2
```

Esto permite representar espacialmente los objetos para sistemas hápticos.

---

# Comunicación serial con Arduino

Las detecciones son enviadas mediante UART serial.

```python
arduino.write((datos_json + "\n").encode('utf-8'))
```

Formato enviado:

```json
[
  {
    "objeto": "car",
    "centroObj": [420.3, 210.8]
  }
]
```

---

# Aplicación háptica

El Arduino puede utilizar las coordenadas recibidas para:

- activar motores vibratorios
- indicar proximidad
- representar dirección espacial
- generar alertas físicas
- construir navegación asistida

---

# Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| YOLO | Detección de objetos |
| COCO Dataset | Modelo base preentrenado |
| OpenCV | Captura y renderizado |
| Python | Lógica principal |
| PySerial | Comunicación serial |
| Arduino | Control físico |
| JSON | Estructura de transmisión |

---

# Estructura del proyecto

```text
modelohaptic/
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
```

---

# Instalación

## Clonar repositorio

```bash
git clone https://github.com/Pollitobello2005/modelohaptic.git
cd modelohaptic
```

---

## Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Dependencias

```txt
ultralytics
opencv-python
pyserial
```

---

# Configuración serial

```python
PUERTO_SERIAL = "COM3"
BAUD_RATE = 9600
```

---

# Ejecución

```bash
python YoloCustom.py
```

Salir:

```text
Presionar tecla Q
```

---

# Objetivo del proyecto

El propósito principal de ModeloHaptic es crear sistemas capaces de interpretar el entorno físico mediante visión artificial y traducir esa información hacia retroalimentación física o háptica.

---

# Futuras mejoras

- Detección de profundidad
- Integración con ESP32
- Seguimiento múltiple de objetos
- Navegación háptica avanzada
- Optimización para edge devices
- Modelos YOLO más ligeros
- Integración con sensores adicionales

---

# Autor

Roberto Carlos Méndez Ortiz  
Ingeniería en Robótica  
Universidad de Guadalajara

---
