"""Leitura e escrita de imagens com OpenCV."""
from pathlib import Path

import cv2
import numpy as np

from . import config


def list_images(folder: Path) -> list[Path]:
    """Lista imagens (recursivamente) de uma pasta, em ordem alfabetica."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray | None:
    """Le uma imagem em BGR. Retorna None se nao for possivel ler."""
    return cv2.imread(str(path))


def save_image(path: Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
