from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return -(probs * np.log(probs + eps)).sum(axis=-1)


@dataclass(frozen=True)
class CalibrationProfile:
    entropy_threshold: float
    aps_quantile: float
    alpha: float
    entropy_quantile: float
    source: str

    @classmethod
    def from_npz(
        cls,
        path: Path,
        *,
        alpha: float = 0.10,
        entropy_quantile: float = 0.90,
    ) -> "CalibrationProfile":
        if not path.exists():
            return cls(
                entropy_threshold=0.50,
                aps_quantile=0.90,
                alpha=alpha,
                entropy_quantile=entropy_quantile,
                source=f"fallback_missing:{path}",
            )

        arrays = np.load(path, allow_pickle=True)
        probs = np.asarray(arrays["known_scores"], dtype=np.float64)
        labels = np.asarray(arrays["y_disease"], dtype=np.int64)
        ent = entropy(probs)
        scores = aps_scores(probs, labels)
        level = min(np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores), 1.0)
        return cls(
            entropy_threshold=float(np.quantile(ent, entropy_quantile)),
            aps_quantile=float(np.quantile(scores, level)),
            alpha=alpha,
            entropy_quantile=entropy_quantile,
            source=str(path),
        )

    def aps_set(self, probs: np.ndarray, valid_indices: Sequence[int]) -> list[int]:
        valid = np.asarray(list(valid_indices), dtype=np.int64)
        valid_probs = probs[valid]
        total = float(valid_probs.sum())
        if total <= 0:
            return [int(valid[int(np.argmax(valid_probs))])]
        valid_probs = valid_probs / total
        order_local = np.argsort(-valid_probs)
        ordered_valid = valid[order_local]
        ordered_probs = valid_probs[order_local]
        cumulative = np.cumsum(ordered_probs)
        cutoff = int(np.searchsorted(cumulative, self.aps_quantile, side="left"))
        cutoff = max(0, min(cutoff, len(ordered_valid) - 1))
        return [int(idx) for idx in ordered_valid[: cutoff + 1]]


def aps_scores(probs: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    out: list[float] = []
    for p, label in zip(probs, labels):
        p = np.asarray(p, dtype=np.float64)
        order = np.argsort(-p)
        cumulative = np.cumsum(p[order])
        rank = np.where(order == int(label))[0]
        if len(rank) == 0:
            out.append(1.0)
        else:
            out.append(float(cumulative[int(rank[0])]))
    return np.asarray(out, dtype=np.float64)
