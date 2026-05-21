# ModeloHaptic

Sistema de visión por computadora y retroalimentación háptica basado en YOLO y Arduino para detección de objetos en tiempo real.

---

# Descripción técnica

ModeloHaptic es un sistema que integra inteligencia artificial, visión por computadora y comunicación serial para detectar objetos mediante una cámara web y transmitir su posición espacial a un microcontrolador Arduino.

El proyecto utiliza un modelo personalizado entrenado con YOLO (You Only Look Once) para identificar objetos dentro de cada frame capturado por la cámara. Posteriormente, calcula el centro geométrico de cada objeto detectado y envía la información en formato JSON a través del puerto serial.

La finalidad del sistema es permitir el desarrollo de interfaces hápticas o sistemas físicos capaces de reaccionar a la posición de objetos en tiempo real.

---

# Arquitectura del sistema

```text
Cámara Web
    ↓
Captura de Frames (OpenCV)
    ↓
Inferencia con YOLO
    ↓
Extracción de Bounding Boxes
    ↓
Cálculo de centro de objetos
    ↓
Serialización JSON
    ↓
Comunicación Serial UART
    ↓
Arduino / Sistema Háptico
```

---

# Flujo de funcionamiento

## 1. Inicialización de conexión serial

El sistema establece comunicación serial con Arduino utilizando PySerial.

```python
arduino = serial.Serial(PUERTO_SERIAL, BAUD_RATE, timeout=1)
```

Esto permite transmitir información de detección hacia dispositivos físicos externos.

---

## 2. Carga del modelo YOLO

Se carga un modelo previamente entrenado utilizando Ultralytics YOLO.

```python
model = YOLO("runs/detect/train/weights/best.pt")
```

El archivo `best.pt` contiene los pesos del modelo entrenado.

---

## 3. Captura de video en tiempo real

OpenCV accede a la cámara principal del sistema.

```python
cap = cv2.VideoCapture(0)
```

Cada iteración del loop obtiene un nuevo frame.

---

## 4. Inferencia de objetos

Cada frame es procesado por YOLO:

```python
results = model(frame)
```

El modelo devuelve:

- Bounding boxes
- Clases detectadas
- Nivel de confianza
- Coordenadas espaciales

---

## 5. Extracción de información espacial

Para cada objeto detectado se obtiene:

- Clase del objeto
- Coordenadas de la caja delimitadora
- Centro geométrico del objeto

```python
x1, y1, x2, y2 = box.xyxy[0].tolist()
```

Cálculo del centro:

```python
centro_x = x1 + (x2 - x1) / 2
centro_y = y1 + (y2 - y1) / 2
```

---

# Estructura de datos transmitida

La información enviada al Arduino tiene formato JSON.

Ejemplo:

```json
[
  {
    "objeto": "persona",
    "centroObj": [320.5, 240.2]
  }
]
```

Esto permite que el microcontrolador interprete fácilmente las posiciones detectadas.

---

# Comunicación serial

Los datos son enviados mediante UART serial:

```python
arduino.write((datos_json + "\n").encode('utf-8'))
```

Parámetros:

| Parámetro | Valor |
|---|---|
| Baud Rate | 9600 |
| Comunicación | UART |
| Encoding | UTF-8 |

---

# Visualización en tiempo real

El sistema muestra las detecciones directamente sobre el video:

```python
annotated_frame = result.plot()
```

Posteriormente se renderiza utilizando OpenCV:

```python
cv2.imshow("YOLO Detección en Vivo", annotated_frame)
```

---

# Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Lógica principal |
| OpenCV | Captura y renderizado de video |
| Ultralytics YOLO | Detección de objetos |
| PySerial | Comunicación serial |
| Arduino | Control físico/háptico |
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

## 1. Clonar repositorio

```bash
git clone https://github.com/Pollitobello2005/modelohaptic.git
cd modelohaptic
```

---

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Dependencias principales

```txt
ultralytics
opencv-python
pyserial
```

---

# Configuración

Modificar el puerto serial según el Arduino conectado:

```python
PUERTO_SERIAL = "COM3"
BAUD_RATE = 9600
```

---

# Ejecución

```bash
python YoloCustom.py
```

Para salir del sistema:

```text
Presionar tecla Q
```

---

# Aplicaciones potenciales

- Sistemas hápticos
- Asistencia para discapacidad visual
- Robótica autónoma
- Interfaces físicas inteligentes
- Sistemas de navegación espacial
- Sensores de proximidad inteligentes
- Interacción humano-máquina

---

# Mejoras futuras

- Detección de profundidad
- Integración con ESP32
- Optimización en edge devices
- Multi-object tracking
- Integración con motores vibratorios
- Modelos YOLO más ligeros
- Sistema de mapeo espacial

---

# Autor

Roberto Carlos Méndez Ortiz  
Ingeniería en Robótica  
Universidad de Guadalajara

---
