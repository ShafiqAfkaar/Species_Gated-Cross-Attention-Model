from __future__ import annotations

import argparse
import os
from pathlib import Path


SAMPLES = [
    (
        "Cat",
        "Ringworm in Cat",
        Path(
            "data/training_data_deduped_splits/seed42/test/Cat/Ringworm in Cat/"
            "02f6a7313a3a__aug_ringworm_in_cat__0_5821.jpg"
        ),
    ),
    (
        "Cattles",
        "Lumpy Skin",
        Path(
            "data/training_data_deduped_splits/seed42/test/Cattles/Lumpy Skin/"
            "00aae99890b3__curated__archive__4___cows_datasets__lumpy__img_.jpg"
        ),
    ),
    (
        "Dog",
        "Fungal Infection in Dog",
        Path(
            "data/training_data_deduped_splits/seed42/test/Dog/Fungal Infection in Dog/"
            "013fe0db1026__curated__archive__train__fungal_infections__fung.jpg"
        ),
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test VetDerm SGCA plus information provider.")
    parser.add_argument("--provider", default="template", choices=["template", "gemini", "ollama"])
    parser.add_argument("--mc-passes", default="5", help="MC-dropout passes for the smoke test.")
    parser.add_argument("--include-gradcam", action="store_true")
    args = parser.parse_args()

    os.environ["VETDERM_INFO_PROVIDER"] = args.provider
    os.environ["VETDERM_MC_PASSES"] = str(args.mc_passes)

    from vetderm_agent.config import get_settings
    from vetderm_agent.inference import VetDermInferenceService
    from vetderm_agent.knowledge import DiseaseCardStore
    from vetderm_agent.llm_providers import get_information_provider

    settings = get_settings()
    service = VetDermInferenceService(settings)
    service.load()
    cards = DiseaseCardStore(settings.disease_cards_path)
    provider = get_information_provider()

    key_available = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    print(f"device={service.device}")
    print(f"provider={provider.__class__.__name__}")
    print(f"gemini_key_available={key_available}")

    all_correct = True
    for species, expected, path in SAMPLES:
        result = service.predict_bytes(
            path.read_bytes(),
            user_species=species,
            include_gradcam=args.include_gradcam,
        )
        if result.get("status") == "unsupported_image":
            all_correct = False
            print()
            print(f"sample={species}")
            print(f"expected={expected}")
            print(f"unsupported_image=True referral={result['referral']['level']} signals={result['referral']['signals']}")
            continue
        top1 = result["top3_diseases"][0]
        correct = top1["disease"] == expected
        all_correct = all_correct and correct
        info = provider.compose(
            cards.get(top1["disease"]),
            suppress=bool(result["referral"]["suppress_information_panel"]),
            referral_banner=str(result["referral"]["banner"]),
        )
        print()
        print(f"sample={species}")
        print(f"expected={expected}")
        print(f"top1={top1['disease']} probability={top1['probability']:.4f} correct={correct}")
        print(f"referral={result['referral']['level']} signals={result['referral']['signals']}")
        print(f"information_provider={info.get('provider', 'template')}")
        print(f"information_error={bool(info.get('llm_error'))}")
    return 0 if all_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
