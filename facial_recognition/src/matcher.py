"""Comparacao de embeddings e decisao de acesso."""
from dataclasses import dataclass

import numpy as np

from .database import AuthorizedEntry


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class MatchResult:
    granted: bool
    person: str | None
    distance: float | None


class FaceMatcher:
    def __init__(self, entries: list[AuthorizedEntry], threshold: float):
        self.entries = entries
        self.threshold = threshold

    def identify(self, embedding: np.ndarray) -> MatchResult:
        if not self.entries:
            return MatchResult(False, None, None)

        best = min(self.entries, key=lambda e: cosine_distance(embedding, e.embedding))
        dist = cosine_distance(embedding, best.embedding)

        if dist <= self.threshold:
            return MatchResult(True, best.person, dist)
        return MatchResult(False, None, dist)
