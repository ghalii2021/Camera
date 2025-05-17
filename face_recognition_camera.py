from deepface import DeepFace
from deepface.modules.verification import find_distance
import cv2
import time
import sys
import pickle
import cvzone
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supprime les logs TensorFlow

# === Initialisation caméra et embeddings ===
cap = cv2.VideoCapture(0)  # Utilisation de la webcam (0 = caméra par défaut)
start_time = time.time()
fps = 0
frame_count = 0
detected_faces = []
last_detection_time = 0

model_name = "Facenet512"
metrics = [{"cosine": 0.30}, {"euclidean": 20.0}, {"euclidean_l2": 0.78}]

ret, frame = cap.read()
if not ret:
    print("Erreur : la caméra ne fournit pas d'image.")
    sys.exit(1)

frame_width = frame.shape[1]
frame_height = frame.shape[0]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_webcam.mp4", fourcc, 20.0, (frame_width, frame_height))

# === Chargement des embeddings ===
try:
    with open(f"./embeddings/embs_facenet512.pkl", "rb") as file:
        embs = pickle.load(file)
        print("Existing embeddings file loaded successfully.")
except FileNotFoundError:
    print("No existing embeddings file found. Check your path.")
    sys.exit(0)

# === Création dossier et CSV ===
os.makedirs("unknown_faces", exist_ok=True)

with open("detections.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Name", "Distance", "Time"])

# === Fonctions utilitaires ===
def calculate_fps(start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
    return fps, current_time

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# === Boucle principale ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : impossible de lire l'image de la caméra.")
        break

    fps, start_time = calculate_fps(start_time)

    if frame_count % 5 == 0:
        detected_faces = []
        results = DeepFace.extract_faces(frame, detector_backend="yolov8", enforce_detection=False)

        for result in results:
            if result["confidence"] >= 0.5:
                fa = result["facial_area"]
                x = fa.get("x", 0)
                y = fa.get("y", 0)
                w = fa.get("w", 0)
                h = fa.get("h", 0)

                x1, y1, x2, y2 = x, y, x + w, y + h
                cropped_face = frame[y:y + h, x:x + w]
                cropped_face_resized = cv2.resize(cropped_face, (224, 224))
                cropped_face_gray = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2GRAY)
                cropped_face_norm = clahe(cropped_face_gray)
                cropped_face_rgb = cv2.cvtColor(cropped_face_norm, cv2.COLOR_GRAY2RGB)

                emb = DeepFace.represent(
                    cropped_face_rgb,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )[0]["embedding"]

                min_dist = float("inf")
                match_name = None

                for name, emb2 in embs.items():
                    dst = find_distance(emb, emb2, list(metrics[2].keys())[0])
                    if dst < min_dist:
                        min_dist = dst
                        match_name = name

                if min_dist < list(metrics[2].values())[0]:
                    detected_faces.append((x1, y1, x2, y2, match_name, min_dist, (0, 255, 0)))
                    print(f"Detected as: {match_name} {min_dist:.2f}")
                else:
                    detected_faces.append((x1, y1, x2, y2, "Inconnu", min_dist, (0, 0, 255)))
                    # Sauvegarde visage inconnu
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"unknown_faces/unknown_{frame_count}_{timestamp}.jpg"
                    cv2.imwrite(filename, cropped_face)

                # Enregistrement CSV
                with open("detections.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        frame_count,
                        match_name if match_name else "Inconnu",
                        round(min_dist, 3),
                        time.strftime("%H:%M:%S", time.localtime())
                    ])

        last_detection_time = frame_count

    # === Affichage ===
    for x1, y1, x2, y2, name, min_dist, color in detected_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cvzone.putTextRect(
            frame,
            f"{name} {min_dist:.2f}",
            (x1 + 10, y1 - 12),
            scale=1.5,
            thickness=2,
            colorR=color,
        )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow("Webcam", frame)
    out.write(frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
