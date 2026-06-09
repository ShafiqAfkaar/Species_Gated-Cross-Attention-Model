from __future__ import annotations

from dataclasses import dataclass, field


NORMAL_DISEASE_NAMES = {
    "cat_normal",
    "dog_normal",
    "normal skin",
}

ZOONOTIC_OR_CONTAGIOUS_KEYWORDS = (
    "ringworm",
    "scabies",
    "mange",
    "sarcoptic",
    "fungal",
)

HERD_HEALTH_OR_REPORTABLE_KEYWORDS = (
    "foot and mouth",
    "lumpy skin",
)


@dataclass(frozen=True)
class ReferralDecision:
    level: str
    banner: str
    signals: list[str] = field(default_factory=list)
    suppress_information_panel: bool = False


def unsupported_image_decision(reason: str = "unsupported_image") -> ReferralDecision:
    return ReferralDecision(
        level="unsupported_image",
        banner=(
            "Unsupported image. This tool is calibrated only for visible cat, cattle or dog skin images. "
            "Upload a clear dermatology photo of one of those species or consult a veterinarian."
        ),
        signals=[reason],
        suppress_information_panel=True,
    )


def evaluate_referral(
    *,
    top1_disease: str,
    top1_probability: float,
    predictive_entropy: float,
    conformal_set_size: int,
    entropy_threshold: float,
    probability_threshold: float = 0.60,
    conformal_ambiguity_probability_threshold: float = 0.95,
    suppress_after_signal_count: int = 2,
    image_quality_warning: str | None = None,
) -> ReferralDecision:
    signals: list[str] = []
    major_signals: list[str] = []
    disease_l = top1_disease.lower()
    is_normal_label = disease_l in NORMAL_DISEASE_NAMES
    force_required = False

    if predictive_entropy > entropy_threshold:
        signals.append("high_predictive_entropy")
        major_signals.append("high_predictive_entropy")
    if conformal_set_size > 1 and top1_probability < conformal_ambiguity_probability_threshold:
        signals.append("ambiguous_conformal_set")
        major_signals.append("ambiguous_conformal_set")
    if top1_probability < probability_threshold:
        signals.append("low_top1_probability")
        major_signals.append("low_top1_probability")
    if not is_normal_label:
        signals.append("non_normal_triage_label")
    if any(keyword in disease_l for keyword in ZOONOTIC_OR_CONTAGIOUS_KEYWORDS):
        signals.append("zoonotic_or_contagious_possible")
        major_signals.append("zoonotic_or_contagious_possible")
    if any(keyword in disease_l for keyword in HERD_HEALTH_OR_REPORTABLE_KEYWORDS):
        signals.append("herd_health_or_reportable_possible")
        major_signals.append("herd_health_or_reportable_possible")
        force_required = True
    if image_quality_warning:
        signals.append(f"image_quality:{image_quality_warning}")
        major_signals.append(f"image_quality:{image_quality_warning}")

    if force_required or len(major_signals) >= suppress_after_signal_count:
        return ReferralDecision(
            level="vet_required",
            banner=(
                "Visit a veterinarian. The model output is uncertain or involves a potentially contagious, zoonotic or herd-health "
                "condition, so this tool should not be used as the next decision step."
            ),
            signals=signals,
            suppress_information_panel=True,
        )

    if signals:
        return ReferralDecision(
            level="vet_recommended",
            banner=(
                "Veterinary review is recommended. This result is an image-based triage signal, not a diagnosis."
            ),
            signals=signals,
            suppress_information_panel=False,
        )

    return ReferralDecision(
        level="educational_only",
        banner=(
            "This tool can show educational information, but it cannot diagnose disease or replace a veterinarian."
        ),
        signals=[],
        suppress_information_panel=False,
    )
