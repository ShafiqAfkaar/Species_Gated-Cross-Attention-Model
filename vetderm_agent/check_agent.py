from __future__ import annotations

import argparse
import getpass
import os
import textwrap
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _find_samples(data_root: Path, *, max_diseases: int | None = None) -> list[tuple[str, str, Path]]:
    samples: list[tuple[str, str, Path]] = []
    for species_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for disease_dir in sorted(path for path in species_dir.iterdir() if path.is_dir()):
            images = sorted(
                path
                for path in disease_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if images:
                samples.append((species_dir.name, disease_dir.name, images[0]))
            if max_diseases is not None and len(samples) >= max_diseases:
                return samples
    return samples


def _format_info(info: dict[str, Any]) -> str:
    if not info.get("shown", False):
        return str(info.get("reason", "Information panel suppressed."))
    if info.get("text"):
        return str(info["text"])
    pieces = [
        str(info.get("overview", "")),
        "Typical signs: " + "; ".join(info.get("typical_signs", [])),
        "Transmission: " + str(info.get("transmission", "")),
        "Ask your vet: " + "; ".join(info.get("ask_your_vet", [])),
        "Urgent flags: " + "; ".join(info.get("urgent_flags", [])),
        str(info.get("safety_note", "")),
    ]
    return "\n".join(piece for piece in pieces if piece.strip())


def _print_wrapped(label: str, text: str, *, width: int = 100) -> None:
    print(label)
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent="  ", initial_indent="  ")
    for paragraph in text.splitlines():
        if paragraph.strip():
            print(wrapper.fill(paragraph.strip()))


def _require_gemini_key() -> None:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return
    key = getpass.getpass("Paste Google AI Studio Gemini API key (input hidden): ").strip()
    if not key:
        raise SystemExit("No API key entered.")
    os.environ["GEMINI_API_KEY"] = key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one image from every Cat/Cattles/Dog disease folder through the SGCA agent."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/training_data_deduped_splits/seed42/test"),
        help="Dataset split containing species/disease/image folders.",
    )
    parser.add_argument("--provider", default="gemini", choices=["gemini", "template", "ollama"])
    parser.add_argument("--model", default=None, help="Gemini or Ollama model override.")
    parser.add_argument("--mc-passes", default="5", help="MC-dropout passes for the check.")
    parser.add_argument("--max-diseases", type=int, default=None, help="Optional quick-test limit.")
    parser.add_argument("--include-gradcam", action="store_true", help="Also generate Grad-CAM files.")
    args = parser.parse_args()

    if args.provider == "gemini":
        _require_gemini_key()
    if args.model and args.provider == "gemini":
        os.environ["VETDERM_GEMINI_MODEL"] = args.model
    if args.model and args.provider == "ollama":
        os.environ["VETDERM_OLLAMA_MODEL"] = args.model

    os.environ["VETDERM_INFO_PROVIDER"] = args.provider
    os.environ["VETDERM_MC_PASSES"] = str(args.mc_passes)

    from vetderm_agent.config import get_settings
    from vetderm_agent.inference import VetDermInferenceService
    from vetderm_agent.knowledge import DiseaseCardStore
    from vetderm_agent.llm_providers import get_information_provider

    settings = get_settings()
    samples = _find_samples(args.data_root, max_diseases=args.max_diseases)
    if not samples:
        raise SystemExit(f"No image samples found under {args.data_root}.")

    service = VetDermInferenceService(settings)
    service.load()
    cards = DiseaseCardStore(settings.disease_cards_path)
    provider = get_information_provider()

    print("VetDerm SGCA agent check")
    print(f"device: {service.device}")
    print(f"provider: {provider.__class__.__name__}")
    print(f"mc_passes: {args.mc_passes}")
    print(f"samples: {len(samples)}")

    correct = 0
    for index, (species, expected, image_path) in enumerate(samples, start=1):
        result = service.predict_bytes(
            image_path.read_bytes(),
            user_species=species,
            include_gradcam=args.include_gradcam,
        )
        if result.get("status") == "unsupported_image":
            print("\n" + "=" * 100)
            print(f"{index:02d}. species={species}")
            print(f"image={image_path}")
            print(f"expected_disease={expected}")
            print(f"unsupported_image=True referral={result['referral']['level']} signals={result['referral']['signals']}")
            continue
        top1 = result["top3_diseases"][0]
        is_correct = top1["disease"] == expected
        correct += int(is_correct)
        info = provider.compose(
            cards.get(top1["disease"]),
            suppress=bool(result["referral"]["suppress_information_panel"]),
            referral_banner=str(result["referral"]["banner"]),
        )

        print("\n" + "=" * 100)
        print(f"{index:02d}. species={species}")
        print(f"image={image_path}")
        print(f"expected_disease={expected}")
        print(f"identified_disease={top1['disease']} probability={top1['probability']:.4f} correct={is_correct}")
        print(f"model_species_prediction={result['model_species_prediction']}")
        print(f"referral={result['referral']['level']} signals={result['referral']['signals']}")
        print(f"information_provider={info.get('provider', 'template')}")
        if info.get("llm_error"):
            print(f"information_error={info['llm_error']}")
        _print_wrapped("disease_information:", _format_info(info))

    print("\n" + "=" * 100)
    print(f"summary: {correct}/{len(samples)} samples matched their folder labels.")
    return 0 if correct == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
