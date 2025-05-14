from deepface import DeepFace
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def crop(input_dir, output_dir, detector_backend="yolov8"):
    # Vérifie si le dossier d'entrée existe
    if not os.path.exists(input_dir):
        print(f"❌ Le dossier d'entrée '{input_dir}' n'existe pas.")
        return

    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    for img_file in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_file)
        img_name = os.path.splitext(img_file)[0]

        try:
            face = DeepFace.extract_faces(
                img_path,
                detector_backend=detector_backend,
                enforce_detection=True,
                grayscale=True,
                target_size=None,
            )[0]["face"]

            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            face = cv2.resize(face, (224, 224))
            output_path = os.path.join(output_dir, f"{img_name}.jpg")
            plt.imsave(output_path, face)
        except Exception as e:
            print(f"⚠️ Erreur lors du traitement de {img_file} : {e}")

# Appel de la fonction
crop("./data", "./cropped_faces")
