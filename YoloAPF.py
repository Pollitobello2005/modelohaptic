import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIGURACIÓN
# =========================
model = YOLO("runs/detect/train/weights/best.pt")

k_att = 1.0
k_rep = 0.8
d0 = 0.25  # radio de influencia (en coords normalizadas)

grid_size = 80  # resolución del heatmap (más bajo = más rápido)


# =========================
# FUNCIONES APF
# =========================
def attractive_potential(X, Y, goal):
    return 0.5 * k_att * ((X - goal[0]) ** 2 + (Y - goal[1]) ** 2)


def repulsive_potential(X, Y, obstacles):
    U_rep = np.zeros_like(X)

    for obs in obstacles:
        dx = X - obs[0]
        dy = Y - obs[1]
        D = np.sqrt(dx**2 + dy**2) + 1e-6  # evitar división por cero

        mask = D <= d0
        U_rep[mask] += 0.5 * k_rep * ((1 / D[mask] - 1 / d0) ** 2)

    return U_rep


def attractive_force(q, goal):
    return -k_att * (q - goal)


def repulsive_force(q, obs):
    d = np.linalg.norm(q - obs) + 1e-6

    if d > d0:
        return np.array([0.0, 0.0])

    return k_rep * (1 / d - 1 / d0) * (1 / (d**3)) * (q - obs)


def total_force(q, goal, obstacles):
    F = attractive_force(q, goal)
    for obs in obstacles:
        F += repulsive_force(q, obs)
    return F


# =========================
# INICIO VIDEO
# =========================
cap = cv2.VideoCapture(0)


# posición del agente (abajo-centro)
def get_agent():
    return np.array([0.5, 0.9])


# objetivo (arriba-centro)
goal = np.array([0.5, 0.1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # =========================
    # YOLO DETECCIÓN
    # =========================
    results = model(frame, stream=True)

    obstacles = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            nx = cx / w
            ny = cy / h

            obstacles.append(np.array([nx, ny]))

            # Dibujos básicos
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    # =========================
    # HEATMAP APF
    # =========================
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    U_att = attractive_potential(X, Y, goal)
    U_rep = repulsive_potential(X, Y, obstacles)

    U = U_att + U_rep

    # normalizar
    U_norm = cv2.normalize(U, None, 0, 255, cv2.NORM_MINMAX)
    U_uint8 = U_norm.astype(np.uint8)

    # aplicar colormap
    heatmap = cv2.applyColorMap(U_uint8, cv2.COLORMAP_JET)

    # redimensionar al frame
    heatmap = cv2.resize(heatmap, (w, h))

    # overlay
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # =========================
    # FUERZA TOTAL (FLECHA)
    # =========================
    q = get_agent()
    F = total_force(q, goal, obstacles)

    agent_px = (int(q[0] * w), int(q[1] * h))
    goal_px = (int(goal[0] * w), int(goal[1] * h))

    # dibujar agente y objetivo
    cv2.circle(overlay, agent_px, 6, (255, 255, 255), -1)
    cv2.circle(overlay, goal_px, 6, (255, 255, 255), -1)

    # flecha
    scale = 200
    end_point = (int(agent_px[0] + F[0] * scale), int(agent_px[1] + F[1] * scale))

    cv2.arrowedLine(overlay, agent_px, end_point, (255, 255, 255), 3)

    # =========================
    # MOSTRAR
    # =========================
    cv2.imshow("YOLO + APF Heatmap", overlay)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
