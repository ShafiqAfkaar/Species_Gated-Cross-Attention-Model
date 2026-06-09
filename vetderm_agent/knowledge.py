from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiseaseCard:
    disease: str
    overview: str
    typical_signs: list[str]
    transmission: str
    ask_your_vet: list[str]
    urgent_flags: list[str]


class DiseaseCardStore:
    def __init__(self, cards_path: Path):
        self.cards_path = cards_path
        self._cards = self._load(cards_path)

    def get(self, disease_name: str) -> DiseaseCard:
        raw = self._cards.get(disease_name)
        if raw is None:
            return DiseaseCard(
                disease=disease_name,
                overview="A curated educational card has not been written for this label yet.",
                typical_signs=["Use the model output only as a visual triage signal."],
                transmission="Transmission information is not available in the local card.",
                ask_your_vet=["Ask whether an in-person skin examination or diagnostic test is needed."],
                urgent_flags=["Rapid worsening, pain, fever, spreading lesions or poor general condition."],
            )
        return DiseaseCard(**raw)

    @staticmethod
    def _load(path: Path) -> dict[str, dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        return {row["disease"]: row for row in rows}


def compose_safe_information(card: DiseaseCard, *, suppress: bool = False) -> dict[str, Any]:
    if suppress:
        return {
            "shown": False,
            "reason": "The information panel was suppressed because the referral logic requires veterinary review first.",
        }
    return {
        "shown": True,
        "disease": card.disease,
        "overview": card.overview,
        "typical_signs": card.typical_signs,
        "transmission": card.transmission,
        "ask_your_vet": card.ask_your_vet,
        "urgent_flags": card.urgent_flags,
        "safety_note": (
            "Educational information only. This tool does not provide treatment, medication, dosing, prognosis or a diagnosis."
        ),
    }
