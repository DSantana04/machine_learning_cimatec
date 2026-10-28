"""Configuracoes centralizadas: caminhos, modelo, limiares e cores."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Pastas
AUTHORIZED_DIR = BASE_DIR / "data" / "authorized"
TEST_DIR = BASE_DIR / "data" / "test"
OUTPUT_DIR = BASE_DIR / "data" / "output"
CACHE_FILE = BASE_DIR / "data" / "cache" / "embeddings.pkl"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# DeepFace
MODEL_NAME = "Facenet"          # Facenet, Facenet512, ArcFace, VGG-Face
DETECTOR_BACKEND = "opencv"     # opencv, ssd, mtcnn, retinaface...

# Limiares de distancia (cosseno) por modelo: abaixo do limiar = mesma pessoa
COSINE_THRESHOLDS = {
    "Facenet": 0.40,
    "Facenet512": 0.30,
    "ArcFace": 0.68,
    "VGG-Face": 0.68,
}

# Visual (OpenCV usa BGR)
COLOR_GRANTED = (0, 200, 0)     # verde
COLOR_DENIED = (0, 0, 255)      # vermelho
TEXT_GRANTED = "acesso liberado"
TEXT_DENIED = "acesso negado"
BOX_THICKNESS = 3
