from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DomainGateDecision:
    enabled: bool
    accepted: bool
    score: float | None = None
    threshold: float | None = None
    reason: str = ""
    nearest_species: str | None = None
    nearest_disease: str | None = None


class DomainGateProfile:
    """Feature-bank gate for rejecting unsupported uploads.

    The profile stores L2-normalised SGCA fused vectors from in-domain training
    images. At inference, an upload is accepted only when its fused vector is
    sufficiently similar to at least one reference image. The threshold is
    calibrated from validation-set nearest-neighbour similarities.
    """

    def __init__(
        self,
        *,
        reference_features: np.ndarray,
        reference_species: np.ndarray,
        reference_diseases: np.ndarray,
        threshold: float,
        source: str,
        chunk_size: int = 4096,
    ):
        features = np.asarray(reference_features, dtype=np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        self.reference_features = features / np.maximum(norms, 1e-8)
        self.reference_species = np.asarray(reference_species)
        self.reference_diseases = np.asarray(reference_diseases)
        self.threshold = float(threshold)
        self.source = source
        self.chunk_size = int(chunk_size)

    @classmethod
    def from_npz(cls, path: Path) -> "DomainGateProfile":
        data = np.load(path, allow_pickle=True)
        return cls(
            reference_features=np.asarray(data["reference_features"], dtype=np.float32),
            reference_species=np.asarray(data["reference_species"]),
            reference_diseases=np.asarray(data["reference_diseases"]),
            threshold=float(np.asarray(data["threshold"]).item()),
            source=str(path),
        )

    def score(self, feature: np.ndarray) -> dict[str, Any]:
        vec = np.asarray(feature, dtype=np.float32).reshape(-1)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
        best_score = -1.0
        best_idx = 0
        for start in range(0, len(self.reference_features), self.chunk_size):
            chunk = self.reference_features[start : start + self.chunk_size]
            sims = chunk @ vec
            local_idx = int(np.argmax(sims))
            local_score = float(sims[local_idx])
            if local_score > best_score:
                best_score = local_score
                best_idx = start + local_idx
        return {
            "score": best_score,
            "nearest_species": str(self.reference_species[best_idx]),
            "nearest_disease": str(self.reference_diseases[best_idx]),
        }

    def decide(self, feature: np.ndarray) -> DomainGateDecision:
        scored = self.score(feature)
        accepted = bool(scored["score"] >= self.threshold)
        return DomainGateDecision(
            enabled=True,
            accepted=accepted,
            score=float(scored["score"]),
            threshold=self.threshold,
            reason="in_domain_feature_similarity" if accepted else "unsupported_image_feature_distance",
            nearest_species=str(scored["nearest_species"]),
            nearest_disease=str(scored["nearest_disease"]),
        )


class DisabledDomainGate:
    source = "disabled"

    def decide(self, feature: np.ndarray) -> DomainGateDecision:
        return DomainGateDecision(enabled=False, accepted=True, reason="domain_gate_disabled")
