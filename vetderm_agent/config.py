from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AgentSettings:
    checkpoint_path: Path = Path(
        os.environ.get(
            "VETDERM_CHECKPOINT",
            PROJECT_ROOT
            / "results"
            / "models"
            / "unified_sgca"
            / "unified_sgca_best.pt",
        )
    )
    calibration_npz: Path = Path(
        os.environ.get(
            "VETDERM_CALIBRATION_NPZ",
            PROJECT_ROOT
            / "figures"
            / "model_comparison"
            / "source_data"
            / "arrays_unified_val.npz",
        )
    )
    disease_cards_path: Path = Path(
        os.environ.get("VETDERM_DISEASE_CARDS", PROJECT_ROOT / "vetderm_agent" / "knowledge" / "disease_cards.json")
    )
    domain_gate_npz: Path = Path(
        os.environ.get(
            "VETDERM_DOMAIN_GATE_NPZ",
            PROJECT_ROOT / "vetderm_agent" / "domain_gate" / "domain_gate_profile.npz",
        )
    )
    domain_gate_enabled: bool = os.environ.get("VETDERM_DOMAIN_GATE_ENABLED", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    gradcam_dir: Path = Path(os.environ.get("VETDERM_GRADCAM_DIR", PROJECT_ROOT / "vetderm_agent" / "static" / "gradcam"))
    audit_log_path: Path = Path(os.environ.get("VETDERM_AUDIT_LOG", PROJECT_ROOT / "vetderm_agent" / "audit" / "inference.jsonl"))
    device: str = os.environ.get("VETDERM_DEVICE", "auto")
    mc_passes: int = int(os.environ.get("VETDERM_MC_PASSES", "30"))
    conformal_alpha: float = float(os.environ.get("VETDERM_CONFORMAL_ALPHA", "0.10"))
    entropy_quantile: float = float(os.environ.get("VETDERM_ENTROPY_QUANTILE", "0.90"))
    probability_threshold: float = float(os.environ.get("VETDERM_PROB_THRESHOLD", "0.60"))
    conformal_ambiguity_probability_threshold: float = float(
        os.environ.get("VETDERM_CONFORMAL_AMBIGUITY_PROB_THRESHOLD", "0.95")
    )
    suppress_info_after_signal_count: int = int(os.environ.get("VETDERM_SUPPRESS_INFO_AFTER_SIGNAL_COUNT", "2"))


def get_settings() -> AgentSettings:
    return AgentSettings()
