# ModeloHaptic

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
