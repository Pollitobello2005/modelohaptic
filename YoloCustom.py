from ultralytics import YOLO
import cv2
import serial
import json
import time
import numpy as np
import os

# Configuración puerto serial
PUERTO_SERIAL = "/dev/ttyACM0"
BAUD_RATE = 9600

# Inicializar conexión con Arduino
arduino = None
try:
    if os.path.exists(PUERTO_SERIAL):
        arduino = serial.Serial(PUERTO_SERIAL, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print(f"Conectado a Arduino en {PUERTO_SERIAL}")
except Exception as e:
    print(f"Error al conectar con Arduino: {e}")

# Categorías de interés y su nivel de peligro
CATEGORIAS_PELIGRO = {
    'car': 'ALTO', 'motorcycle': 'ALTO', 'bus': 'ALTO', 'train': 'ALTO', 'truck': 'ALTO',
    'bicycle': 'MEDIO', 'person': 'MEDIO', 'bench': 'MEDIO', 'fire hydrant': 'MEDIO',
    'traffic light': 'BAJO', 'stop sign': 'BAJO'
}

# Valores base para la Ecuación de Riesgo Dinámico
VALOR_BASE_RIESGO = {'ALTO': 20, 'MEDIO': 10, 'BAJO': 0}

ALTURAS_REALES = {
    'car': 1.5, 'motorcycle': 1.2, 'bus': 3.0, 'train': 3.5, 'truck': 3.0,
    'bicycle': 1.0, 'person': 1.7, 'bench': 0.8, 'fire hydrant': 0.8,
    'traffic light': 0.6, 'stop sign': 0.8
}
FOCAL_LENGTH = 600.0

# Cargar modelo de Segmentación YOLOv8
model = YOLO("yolov8n-seg.pt")

# Física APF
k_att_global = 1.0
d0_rep_norm = 0.25 
arrow_scale_visual = 200.0

def calc_repulsive_force_norm(q_norm, obs_norm, k_rep, d0):
    d = np.linalg.norm(q_norm - obs_norm) + 1e-6
    if d > d0: 
        return np.array([0.0, 0.0])
    direction = q_norm - obs_norm
    direction = direction / d
    mag = k_rep * (1.0 / d - 1.0 / d0) * (1.0 / (d**2))
    return mag * direction
    
def calc_attractive_force_norm(q_norm, goal_norm, k_att):
    return k_att * (goal_norm - q_norm)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cv2.namedWindow("HAPTIC APF VISION", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret: break

    h_f, w_f = frame.shape[:2]
    area_total_pantalla = w_f * h_f

    results = model(frame, verbose=False)
    detected_objects = []
    obstacles_apf = []  
    
    annotated_frame = frame.copy()
    overlay = frame.copy()
    
    for result in results:
        if result.masks is None or result.boxes is None:
            continue
            
        boxes = result.boxes
        masks = result.masks
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            
            if name not in CATEGORIAS_PELIGRO: continue
                
            peligro_base = CATEGORIAS_PELIGRO[name]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_f, x2), min(h_f, y2)
            if x2 <= x1 or y2 <= y1: continue
            
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            area_obj = (x2 - x1) * (y2 - y1)
            ratio_area = area_obj / area_total_pantalla
            
            puntos_proximidad = min(ratio_area * 100, 80.0) 
            puntos_base = VALOR_BASE_RIESGO[peligro_base]
            score_total = puntos_base + puntos_proximidad
            
            # Umbrales de color para los recuadros
            if score_total > 75: 
                color_bgr = (0, 0, 255)      # Rojo
                nivel_prioridad = "CRITICO"
                k_rep_val = 6.0
            elif score_total > 45: 
                color_bgr = (0, 128, 255)    # Naranja
                nivel_prioridad = "ALTO"
                k_rep_val = 4.0
            elif score_total > 15: 
                color_bgr = (0, 255, 255)    # Amarillo
                nivel_prioridad = "MEDIO"
                k_rep_val = 2.0
            else:
                color_bgr = (0, 255, 0)      # Verde
                nivel_prioridad = "BAJO"
                k_rep_val = 0.5

            detected_objects.append({
                "objeto": name,
                "score": round(score_total, 1),
                "nivel": nivel_prioridad,
                "centroObj": [cx, cy]
            })
            
            d0_local = max(x2 - x1, y2 - y1) / 1.0  
            Y_grid, X_grid = np.mgrid[y1:y2, x1:x2]
            D = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
            D_safe = np.clip(D, 1.0, None)
            
            U_rep = np.zeros_like(D, dtype=np.float32)
            infl = D_safe <= d0_local
            U_rep[infl] = 0.5 * k_rep_val * ((1.0 / D_safe[infl]) - (1.0 / d0_local))**2
            
            U_norm = cv2.normalize(U_rep, None, 0, 1.0, cv2.NORM_MINMAX)
            
            heat_roi = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
            heat_roi[:, :, 0] = (color_bgr[0] * U_norm).astype(np.uint8)
            heat_roi[:, :, 1] = (color_bgr[1] * U_norm).astype(np.uint8)
            heat_roi[:, :, 2] = (color_bgr[2] * U_norm).astype(np.uint8)
            
            segment = masks.xy[i]
            if len(segment) > 0:
                pts = np.int32([segment])
                obj_mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(obj_mask_full, pts, 255)
                
                obj_mask_roi = obj_mask_full[y1:y2, x1:x2] == 255
                
                overlay_roi = overlay[y1:y2, x1:x2]
                overlay_roi[obj_mask_roi] = heat_roi[obj_mask_roi]
                overlay[y1:y2, x1:x2] = overlay_roi
                
                h_px = y2 - y1
                h_real = ALTURAS_REALES.get(name, 1.0)
                dist_m = (h_real * FOCAL_LENGTH) / max(1, h_px)
                
                obstacles_apf.append({
                    'center': np.array([cx, cy], dtype=np.float32), 
                    'k_rep': k_rep_val,
                    'radius': int(d0_rep_norm * h_f),
                    'bbox': (x1, y1, x2, y2),
                    'objeto': name,
                    'score': score_total,
                    'color': color_bgr,
                    'distancia': dist_m
                })

    detected_objects.sort(key=lambda x: x['score'], reverse=True)
    obstacles_apf.sort(key=lambda x: x['score'], reverse=True)

    alpha = 0.65
    annotated_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    for obs in obstacles_apf:
        cx, cy = int(obs['center'][0]), int(obs['center'][1])
        x1, y1, x2, y2 = obs['bbox']
        color = obs['color']
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{obs['objeto']} | {obs['distancia']:.1f}m | PTS: {int(obs['score'])}"
        cv2.putText(annotated_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(annotated_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    agent_px = np.array([w_f / 2.0, h_f * 0.75])
    goal_px = np.array([w_f / 2.0, h_f * 0.25])
    agent_norm = agent_px / np.array([w_f, h_f])
    goal_norm = goal_px / np.array([w_f, h_f])
    
    F_tot_norm = calc_attractive_force_norm(agent_norm, goal_norm, k_att_global)
    score_maximo = obstacles_apf[0]['score'] if obstacles_apf else 0
    
    for obs in obstacles_apf:
        obs_norm = obs['center'] / np.array([w_f, h_f])
        F_tot_norm += calc_repulsive_force_norm(agent_norm, obs_norm, obs['k_rep'], d0_rep_norm)
            
    F_tot_norm = np.clip(F_tot_norm, -2.0, 2.0)
    end_pt_x = np.clip(int(agent_px[0] + F_tot_norm[0] * arrow_scale_visual), 30, w_f - 30)
    end_pt_y = np.clip(int(agent_px[1] + F_tot_norm[1] * arrow_scale_visual), 30, h_f - 30)
    
    f_x, f_y = F_tot_norm[0], F_tot_norm[1]
    
    # ========================================================
    # LÓGICA DE ALERTA HUD (Sincronizada estrictamente con el recuadro)
    # ========================================================
    if score_maximo > 75: 
        # Si hay un recuadro ROJO en pantalla
        hud_text = "!!! FRENE: OBSTACULO INMINENTE !!!"
        hud_color = (0, 0, 255) 
        hud_level = 2
    elif score_maximo > 45: 
        # Si el peor recuadro en pantalla es NARANJA
        dir_txt = "DERECHA ->" if f_x > 0 else "<- IZQUIERDA"
        hud_text = f"EVADA A LA {dir_txt}"
        hud_color = (0, 128, 255) 
        hud_level = 1
    elif score_maximo > 15: 
        # Si el peor recuadro en pantalla es AMARILLO
        hud_text = "PRECAUCION FRENTE"
        hud_color = (0, 255, 255) 
        hud_level = 1
    else:
        # Puros recuadros VERDES o pantalla limpia
        hud_text = "VIA LIBRE"
        hud_color = (0, 255, 0) 
        hud_level = 0

    # RENDERIZADO UI
    if hud_level > 0:
        vig_overlay = np.zeros_like(annotated_frame)
        cv2.rectangle(vig_overlay, (0,0), (w_f, h_f), hud_color, 15)
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, vig_overlay, 0.3, 0)

    centro_tuple = (int(agent_px[0]), int(agent_px[1]))
    fin_tuple = (int(end_pt_x), int(end_pt_y))
    cv2.arrowedLine(annotated_frame, centro_tuple, fin_tuple, hud_color, 10, tipLength=0.2)
    cv2.arrowedLine(annotated_frame, centro_tuple, fin_tuple, (255, 255, 255), 2, tipLength=0.2)
    
    panel_height = 70
    panel_overlay = annotated_frame.copy()
    cv2.rectangle(panel_overlay, (0, h_f - panel_height), (w_f, h_f), (15, 15, 18), -1)
    cv2.line(panel_overlay, (0, h_f - panel_height), (w_f, h_f - panel_height), hud_color, 2)
    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, panel_overlay, 0.3, 0)
    
    text_size = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    txt_x = int((w_f - text_size[0]) / 2)
    txt_y = int(h_f - (panel_height / 2) + (text_size[1] / 2))
    
    cv2.putText(annotated_frame, hud_text, (txt_x + 1, txt_y + 1), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, hud_text, (txt_x, txt_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, hud_color, 2, cv2.LINE_AA)
    
    cv2.imshow("HAPTIC APF VISION", annotated_frame)
    
    if arduino and arduino.is_open:
        try:
            payload = {"hud": hud_text, "objects": detected_objects}
            datos_json = json.dumps(payload)
            arduino.write((datos_json + "\n").encode('utf-8'))
        except Exception: pass
    
    if cv2.waitKey(10) & 0xFF == ord('q'): break
    
cap.release()
cv2.destroyAllWindows()
if arduino and arduino.is_open:
    arduino.close()
