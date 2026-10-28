"""Desenho das bounding boxes (verde = liberado, vermelho = negado) com OpenCV."""
import unicodedata

import cv2
import numpy as np

from . import config
from .face_engine import DetectedFace
from .matcher import MatchResult


def _ascii(text: str) -> str:
    """cv2.putText nao renderiza acentos; normaliza para ASCII."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def draw_decision(image: np.ndarray, face: DetectedFace, result: MatchResult) -> None:
    """Desenha a caixa e o texto diretamente na imagem (in-place)."""
    color = config.COLOR_GRANTED if result.granted else config.COLOR_DENIED
    label = config.TEXT_GRANTED if result.granted else config.TEXT_DENIED
    if result.granted and result.person:
        label = f"{label} - {_ascii(result.person)}"

    x, y, w, h = face.x, face.y, face.w, face.h
    cv2.rectangle(image, (x, y), (x + w, y + h), color, config.BOX_THICKNESS)

    scale = max(0.5, min(image.shape[1] / 1000, 1.2))
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    top = y - th - baseline - 8
    if top < 0:  # sem espaco acima: coloca dentro da caixa
        top = y
    cv2.rectangle(image, (x, top), (x + tw + 8, top + th + baseline + 8), color, -1)
    cv2.putText(image, label, (x + 4, top + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
