"""Configurações e constantes globais do projeto."""
from pathlib import Path

# Caminhos
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "cnn_cifar10.keras"

# Dados
CLASS_NAMES = [
    "avião", "automóvel", "pássaro", "gato", "cervo",
    "cachorro", "sapo", "cavalo", "navio", "caminhão",
]
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = len(CLASS_NAMES)

# Treinamento
EPOCHS = 50
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 3