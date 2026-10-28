"""Deteccao de rostos e extracao de embeddings via DeepFace."""
from dataclasses import dataclass

import numpy as np
from deepface import DeepFace

from . import config


@dataclass
class DetectedFace:
    x: int
    y: int
    w: int
    h: int
    embedding: np.ndarray

    @property
    def area(self) -> int:
        return self.w * self.h


class FaceEngine:
    """Encapsula o DeepFace: detecta rostos e gera embeddings."""

    def __init__(self, model_name: str = config.MODEL_NAME,
                 detector_backend: str = config.DETECTOR_BACKEND):
        self.model_name = model_name
        self.detector_backend = detector_backend

    def extract_faces(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Retorna todos os rostos encontrados com bounding box e embedding."""
        try:
            results = DeepFace.represent(
                img_path=image_bgr,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True,
            )
        except ValueError:
            # DeepFace levanta ValueError quando nenhum rosto e detectado
            return []

        faces = []
        for r in results:
            area = r["facial_area"]
            faces.append(DetectedFace(
                x=int(area["x"]), y=int(area["y"]),
                w=int(area["w"]), h=int(area["h"]),
                embedding=np.asarray(r["embedding"], dtype=np.float32),
            ))
        return faces

    def extract_main_face(self, image_bgr: np.ndarray) -> DetectedFace | None:
        """Retorna apenas o maior rosto (usado no cadastro de autorizados)."""
        faces = self.extract_faces(image_bgr)
        return max(faces, key=lambda f: f.area) if faces else None
