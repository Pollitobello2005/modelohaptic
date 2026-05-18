from ultralytics import YOLO
import cv2
import serial
import json
import time

# Configuración puerto serial
PUERTO_SERIAL = "COM3"
BAUD_RATE = 9600

# Inicializar conexión con Arduino
try:
    arduino = serial.Serial(PUERTO_SERIAL, BAUD_RATE, timeout=1)
    time.sleep(2)  # Esperar a que Arduino se inicialice
    print(f"Conectado a Arduino en {PUERTO_SERIAL} a {BAUD_RATE} baud")
except Exception as e:
    print(f"Error al conectar con Arduino: {e}")
    arduino = None

# Cargar modelo entrenado
model = YOLO("runs/detect/train/weights/best.pt")
# Iniciar camara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer frame")
        break
    # Detecccion en el frame actual
    results = model(frame)
    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detected_objects.append({"objeto": name,"centroObj": [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]})
        # Enmarca lo detectado
        annotated_frame = result.plot()
    
    # Enviar detecciones a Arduino
    if arduino and arduino.is_open:
        try:
            datos_json = json.dumps(detected_objects)
            arduino.write((datos_json + "\n").encode('utf-8'))
        except Exception as e:
            print(f"Error al enviar datos a Arduino: {e}")
    
    # Video en vivo
    cv2.imshow("YOLO Detección en Vivo", annotated_frame)
    # Detecciones del frame
    print(detected_objects)
    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

# Cerrar conexión serial
if arduino and arduino.is_open:
    arduino.close()
    print("Conexión con Arduino cerrada")