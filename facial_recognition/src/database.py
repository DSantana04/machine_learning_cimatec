"""Base de pessoas autorizadas, com cache de embeddings."""
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .face_engine import FaceEngine
from .image_io import list_images, read_image


@dataclass
class AuthorizedEntry:
    person: str
    image_path: Path
    embedding: np.ndarray


class AuthorizedDatabase:
    """
    Carrega fotos da pasta de autorizados.

    Convencao de nomes:
      - data/authorized/<nome_da_pessoa>/foto1.jpg  -> pessoa = nome da subpasta
      - data/authorized/<nome_da_pessoa>.jpg        -> pessoa = nome do arquivo
    """

    def __init__(self, engine: FaceEngine,
                 authorized_dir: Path = config.AUTHORIZED_DIR,
                 cache_file: Path = config.CACHE_FILE,
                 use_cache: bool = True):
        self.engine = engine
        self.authorized_dir = Path(authorized_dir)
        self.cache_file = Path(cache_file)
        self.use_cache = use_cache

    def _person_name(self, image_path: Path) -> str:
        if image_path.parent == self.authorized_dir:
            return image_path.stem
        return image_path.parent.name

    def _cache_key(self, image_path: Path) -> str:
        rel = image_path.relative_to(self.authorized_dir)
        return f"{self.engine.model_name}|{self.engine.detector_backend}|{rel}|{image_path.stat().st_mtime_ns}"

    def _load_cache(self) -> dict:
        if self.use_cache and self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache: dict) -> None:
        if not self.use_cache:
            return
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache, f)

    def load(self) -> list[AuthorizedEntry]:
        cache = self._load_cache()
        new_cache = {}
        entries: list[AuthorizedEntry] = []

        for path in list_images(self.authorized_dir):
            key = self._cache_key(path)
            embedding = cache.get(key)

            if embedding is None:
                image = read_image(path)
                face = self.engine.extract_main_face(image) if image is not None else None
                if face is None:
                    print(f"[aviso] nenhum rosto detectado em {path.name}, ignorado")
                    continue
                embedding = face.embedding

            new_cache[key] = embedding
            entries.append(AuthorizedEntry(self._person_name(path), path, embedding))

        self._save_cache(new_cache)
        return entries
