"""Orquestra o fluxo: carrega autorizados, processa imagens de teste e gera saidas."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .database import AuthorizedDatabase
from .face_engine import FaceEngine
from .image_io import list_images, read_image, save_image
from .matcher import FaceMatcher
from .visualizer import draw_decision


@dataclass
class FaceReport:
    image: str
    face_index: int
    decision: str
    person: str
    distance: str


class AccessControlSystem:
    def __init__(self, model_name: str = config.MODEL_NAME,
                 detector_backend: str = config.DETECTOR_BACKEND,
                 threshold: float | None = None,
                 authorized_dir: Path = config.AUTHORIZED_DIR,
                 use_cache: bool = True):
        self.engine = FaceEngine(model_name, detector_backend)
        self.threshold = threshold if threshold is not None else config.COSINE_THRESHOLDS[model_name]
        db = AuthorizedDatabase(self.engine, authorized_dir, use_cache=use_cache)
        self.entries = db.load()
        self.matcher = FaceMatcher(self.entries, self.threshold)

    def process_image(self, image: np.ndarray) -> tuple[np.ndarray, list]:
        """Analisa uma imagem BGR e devolve (imagem anotada, lista de (face, resultado))."""
        annotated = image.copy()
        decisions = []
        for face in self.engine.extract_faces(image):
            result = self.matcher.identify(face.embedding)
            draw_decision(annotated, face, result)
            decisions.append((face, result))
        return annotated, decisions

    def process_folder(self, test_dir: Path, output_dir: Path) -> list[FaceReport]:
        reports: list[FaceReport] = []
        for path in list_images(test_dir):
            image = read_image(path)
            if image is None:
                print(f"[aviso] nao foi possivel ler {path.name}")
                continue

            annotated, decisions = self.process_image(image)
            save_image(Path(output_dir) / f"{path.stem}_resultado.jpg", annotated)

            if not decisions:
                print(f"{path.name}: nenhum rosto detectado")
                reports.append(FaceReport(path.name, 0, "sem rosto", "-", "-"))
                continue

            for i, (_, r) in enumerate(decisions, start=1):
                label = config.TEXT_GRANTED if r.granted else config.TEXT_DENIED
                dist = f"{r.distance:.4f}" if r.distance is not None else "-"
                print(f"{path.name} [rosto {i}]: {label} | pessoa={r.person or '-'} | distancia={dist}")
                reports.append(FaceReport(path.name, i, label, r.person or "-", dist))
        return reports

    @staticmethod
    def write_report(reports: list[FaceReport], output_dir: Path) -> Path:
        out = Path(output_dir) / "relatorio.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["imagem", "rosto", "decisao", "pessoa", "distancia"])
            for r in reports:
                writer.writerow([r.image, r.face_index, r.decision, r.person, r.distance])
        return out
