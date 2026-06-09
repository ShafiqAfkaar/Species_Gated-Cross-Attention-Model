from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sgca.models import (
    CLASS_TO_SPECIES,
    GradCAMExporter,
    build_eval_transform,
    load_checkpoint,
    save_gradcam_artifacts,
)

from .audit import append_audit_record
from .calibration import CalibrationProfile, entropy
from .config import AgentSettings, get_settings
from .domain_gate import DisabledDomainGate, DomainGateProfile
from .image_quality import basic_image_quality_warning
from .safety import evaluate_referral, unsupported_image_decision


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def valid_disease_indices(species: str, disease_names: list[str]) -> list[int]:
    out = [idx for idx, disease in enumerate(disease_names) if CLASS_TO_SPECIES.get(disease) == species]
    if not out:
        raise ValueError(f"No valid diseases are configured for species={species!r}")
    return out


def renormalize_to_indices(probs: np.ndarray, indices: list[int]) -> np.ndarray:
    conditioned = np.zeros_like(probs, dtype=np.float64)
    total = float(np.asarray(probs)[indices].sum())
    if total <= 0:
        conditioned[indices] = 1.0 / len(indices)
    else:
        conditioned[indices] = np.asarray(probs, dtype=np.float64)[indices] / total
    return conditioned


class VetDermInferenceService:
    def __init__(self, settings: AgentSettings | None = None):
        self.settings = settings or get_settings()
        self.device = choose_device(self.settings.device)
        self.model: nn.Module | None = None
        self.ckpt: dict[str, Any] | None = None
        self.species_names: list[str] = []
        self.disease_names: list[str] = []
        self.transform = None
        self.calibration = CalibrationProfile.from_npz(
            self.settings.calibration_npz,
            alpha=self.settings.conformal_alpha,
            entropy_quantile=self.settings.entropy_quantile,
        )
        self.domain_gate = DisabledDomainGate()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.settings.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.settings.checkpoint_path}")
        model, ckpt = load_checkpoint(self.settings.checkpoint_path, pretrained=False, device=self.device)
        self.model = model
        self.ckpt = ckpt
        self.species_names = list(ckpt["species_names"])
        self.disease_names = list(ckpt["disease_names"])
        self.transform = build_eval_transform(int(ckpt["img_size"]))
        if self.settings.domain_gate_enabled and self.settings.domain_gate_npz.exists():
            self.domain_gate = DomainGateProfile.from_npz(self.settings.domain_gate_npz)
        self.settings.gradcam_dir.mkdir(parents=True, exist_ok=True)

    def predict_bytes(
        self,
        image_bytes: bytes,
        *,
        user_species: str,
        mc_passes: int | None = None,
        include_gradcam: bool = True,
    ) -> dict[str, Any]:
        self.load()
        assert self.model is not None
        assert self.transform is not None

        if user_species not in self.species_names:
            raise ValueError(f"user_species must be one of {self.species_names}; got {user_species!r}")

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image_quality_warning = basic_image_quality_warning(image_bytes)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        valid_indices = valid_disease_indices(user_species, self.disease_names)

        species_probs, raw_probs, fused_vec = self._deterministic_outputs(tensor)
        domain_decision = self.domain_gate.decide(fused_vec)
        if domain_decision.enabled and not domain_decision.accepted:
            referral = unsupported_image_decision(domain_decision.reason)
            result = {
                "image_hash": image_hash,
                "status": "unsupported_image",
                "disclaimer": "Decision-support and education only. This output is not a veterinary diagnosis or treatment plan.",
                "user_species": user_species,
                "model_species_prediction": self.species_names[int(np.argmax(species_probs))],
                "model_species_confidence": float(np.max(species_probs)),
                "top3_diseases": [],
                "conformal_prediction_set": [],
                "uncertainty": {
                    "mc_passes": 0,
                    "predictive_entropy": None,
                    "mutual_information": None,
                    "entropy_threshold": self.calibration.entropy_threshold,
                    "calibration_source": self.calibration.source,
                    "raw_disease_top1_probability": float(np.max(raw_probs)),
                },
                "domain_gate": domain_decision.__dict__,
                "referral": {
                    "level": referral.level,
                    "banner": referral.banner,
                    "signals": referral.signals,
                    "suppress_information_panel": referral.suppress_information_panel,
                },
                "image_quality_warning": image_quality_warning,
                "gradcam_artifacts": None,
            }
            append_audit_record(
                self.settings.audit_log_path,
                {
                    "image_hash": image_hash,
                    "user_species": user_species,
                    "status": "unsupported_image",
                    "top1_disease": "",
                    "top1_probability": None,
                    "domain_gate_score": domain_decision.score,
                    "domain_gate_threshold": domain_decision.threshold,
                    "referral_level": referral.level,
                    "referral_signals": referral.signals,
                },
            )
            return result

        mc = self._mc_probs(tensor, passes=mc_passes or self.settings.mc_passes)
        disease_probs = renormalize_to_indices(mc["mean_probs"], valid_indices)
        pass_conditioned = np.stack([renormalize_to_indices(p, valid_indices) for p in mc["all_passes"]], axis=0)
        pred_entropy = float(entropy(disease_probs))
        pass_entropy = entropy(pass_conditioned)
        mutual_info = float(max(pred_entropy - float(pass_entropy.mean()), 0.0))
        top3 = self._topk(disease_probs, k=min(3, len(valid_indices)))
        top1 = top3[0]
        conformal_indices = self.calibration.aps_set(disease_probs, valid_indices)
        conformal_set = [
            {"disease": self.disease_names[idx], "probability": float(disease_probs[idx])}
            for idx in conformal_indices
        ]

        referral = evaluate_referral(
            top1_disease=top1["disease"],
            top1_probability=top1["probability"],
            predictive_entropy=pred_entropy,
            conformal_set_size=len(conformal_set),
            entropy_threshold=self.calibration.entropy_threshold,
            probability_threshold=self.settings.probability_threshold,
            conformal_ambiguity_probability_threshold=self.settings.conformal_ambiguity_probability_threshold,
            suppress_after_signal_count=self.settings.suppress_info_after_signal_count,
            image_quality_warning=image_quality_warning,
        )

        gradcam_artifacts: dict[str, str] | None = None
        if include_gradcam:
            gradcam_artifacts = self._gradcam(tensor, class_idx=int(top1["index"]), image_hash=image_hash)

        result = {
            "image_hash": image_hash,
            "status": "ok",
            "disclaimer": "Decision-support and education only. This output is not a veterinary diagnosis or treatment plan.",
            "user_species": user_species,
            "model_species_prediction": self.species_names[int(np.argmax(species_probs))],
            "model_species_confidence": float(np.max(species_probs)),
            "top3_diseases": top3,
            "conformal_prediction_set": conformal_set,
            "uncertainty": {
                "mc_passes": int(mc["passes"]),
                "predictive_entropy": pred_entropy,
                "mutual_information": mutual_info,
                "entropy_threshold": self.calibration.entropy_threshold,
                "calibration_source": self.calibration.source,
                "raw_disease_top1_probability": float(np.max(raw_probs)),
            },
            "domain_gate": domain_decision.__dict__,
            "referral": {
                "level": referral.level,
                "banner": referral.banner,
                "signals": referral.signals,
                "suppress_information_panel": referral.suppress_information_panel,
            },
            "image_quality_warning": image_quality_warning,
            "gradcam_artifacts": gradcam_artifacts,
        }

        append_audit_record(
            self.settings.audit_log_path,
            {
                "image_hash": image_hash,
                "user_species": user_species,
                "top1_disease": top1["disease"],
                "top1_probability": top1["probability"],
                "predictive_entropy": pred_entropy,
                "mutual_information": mutual_info,
                "conformal_set_size": len(conformal_set),
                "referral_level": referral.level,
                "referral_signals": referral.signals,
            },
        )
        return result

    @torch.no_grad()
    def _deterministic_outputs(self, tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.model is not None
        self.model.eval()
        if hasattr(self.model, "forward_features"):
            species_logits, disease_logits, _, fused_vec, _ = self.model.forward_features(tensor, return_attention=False)
        else:
            species_logits, disease_logits = self.model(tensor)
            fused_vec = disease_logits
        species_probs = F.softmax(species_logits, dim=1).detach().cpu().numpy()[0]
        disease_probs = F.softmax(disease_logits, dim=1).detach().cpu().numpy()[0]
        fused = fused_vec.detach().cpu().numpy()[0]
        return species_probs, disease_probs, fused

    @torch.no_grad()
    def _mc_probs(self, tensor: torch.Tensor, *, passes: int) -> dict[str, Any]:
        assert self.model is not None
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        all_passes = []
        for _ in range(passes):
            _, disease_logits = self.model(tensor)
            all_passes.append(F.softmax(disease_logits, dim=1).detach().cpu().numpy()[0])
        self.model.eval()
        stacked = np.stack(all_passes, axis=0)
        return {"passes": passes, "all_passes": stacked, "mean_probs": stacked.mean(axis=0)}

    def _topk(self, probs: np.ndarray, *, k: int) -> list[dict[str, Any]]:
        order = np.argsort(-probs)[:k]
        return [
            {
                "rank": rank + 1,
                "index": int(idx),
                "disease": self.disease_names[int(idx)],
                "probability": float(probs[int(idx)]),
            }
            for rank, idx in enumerate(order)
        ]

    def _gradcam(self, tensor: torch.Tensor, *, class_idx: int, image_hash: str) -> dict[str, str] | None:
        assert self.model is not None
        try:
            self.model.eval()
            exporter = GradCAMExporter(self.model)
            heatmap = exporter.generate(tensor, class_idx=class_idx)
            prefix = self.settings.gradcam_dir / image_hash[:16]
            return save_gradcam_artifacts(tensor.squeeze(0), heatmap, prefix)
        except Exception as exc:  # Grad-CAM should not block triage output.
            return {"error": str(exc)}
