#!/usr/bin/env python3
"""Evaluate VetDerm agent behaviour on wrong-domain OOD images.

For each OOD image, the script simulates the required species dropdown by
running the upload under Cat, Cattles and Dog. Because OOD images have no true
veterinary disease label, the primary endpoint is escalation:

    safe_escalation = referral != educational_only

Educational-only outputs on wrong-domain images are counted as unflagged OOD
outputs and should be inspected.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPECIES = ["Cat", "Cattles", "Dog"]


def find_images(image_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return images[:limit] if limit is not None else images


def flag_dict(signals: list[str]) -> dict[str, bool]:
    return {
        "flag_entropy": "high_predictive_entropy" in signals,
        "flag_conformal": "ambiguous_conformal_set" in signals,
        "flag_low_probability": "low_top1_probability" in signals,
        "flag_non_normal": "non_normal_triage_label" in signals,
        "flag_zoonotic": "zoonotic_or_contagious_possible" in signals,
        "flag_herd_health": "herd_health_or_reportable_possible" in signals,
        "flag_image_quality": any(signal.startswith("image_quality:") for signal in signals),
    }


def major_signal_count(flags: dict[str, bool]) -> int:
    return int(
        flags["flag_entropy"]
        + flags["flag_conformal"]
        + flags["flag_low_probability"]
        + flags["flag_zoonotic"]
        + flags["flag_herd_health"]
        + flags["flag_image_quality"]
    )


def write_summary_tables(df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    referral_order = ["educational_only", "vet_recommended", "vet_required", "unsupported_image"]
    rows = []
    for species, part in [("all", df), *df.groupby("user_species", sort=False)]:
        n = len(part)
        row = {"user_species": species, "n": n}
        for referral in referral_order:
            row[referral] = int((part["referral"] == referral).sum())
            row[f"{referral}_rate"] = float((part["referral"] == referral).mean()) if n else 0.0
        row["safe_escalation"] = int((part["referral"] != "educational_only").sum())
        row["safe_escalation_rate"] = float((part["referral"] != "educational_only").mean()) if n else 0.0
        row["unflagged_ood"] = int((part["referral"] == "educational_only").sum())
        row["unflagged_ood_rate"] = float((part["referral"] == "educational_only").mean()) if n else 0.0
        row["median_p_max"] = float(part["p_max"].median()) if n else 0.0
        row["median_entropy"] = float(part["entropy"].median()) if n else 0.0
        rows.append(row)
    referral_summary = pd.DataFrame(rows)
    referral_summary.to_csv(out_dir / "ood_referral_summary.csv", index=False)

    signal_rows = []
    signal_cols = [
        "flag_entropy",
        "flag_conformal",
        "flag_low_probability",
        "flag_non_normal",
        "flag_zoonotic",
        "flag_herd_health",
        "flag_image_quality",
    ]
    for species, part in [("all", df), *df.groupby("user_species", sort=False)]:
        for column in signal_cols:
            signal_rows.append(
                {
                    "user_species": species,
                    "signal": column,
                    "n": int(part[column].sum()),
                    "rate": float(part[column].mean()) if len(part) else 0.0,
                }
            )
    signal_summary = pd.DataFrame(signal_rows)
    signal_summary.to_csv(out_dir / "ood_signal_summary.csv", index=False)

    label_rows = []
    for (species, label), part in df.groupby(["user_species", "pred_disease"], sort=False):
        label_rows.append(
            {
                "user_species": species,
                "pred_disease": label,
                "n": int(len(part)),
                "rate_within_species": float(len(part) / len(df[df["user_species"] == species])),
                "median_p_max": float(part["p_max"].median()),
                "referral_levels": "|".join(f"{k}:{v}" for k, v in Counter(part["referral"]).most_common()),
            }
        )
    label_summary = pd.DataFrame(label_rows).sort_values(["user_species", "n"], ascending=[True, False])
    label_summary.to_csv(out_dir / "ood_predicted_label_summary.csv", index=False)

    return {
        "referral": referral_summary,
        "signals": signal_summary,
        "labels": label_summary,
    }


def make_failure_contact_sheet(df: pd.DataFrame, out_path: Path, *, max_images: int = 24) -> None:
    failures = df[df["referral"] == "educational_only"].drop_duplicates("path").head(max_images)
    if failures.empty:
        return

    thumb = (150, 150)
    pad = 10
    label_h = 54
    cols = 4
    rows = int((len(failures) + cols - 1) // cols)
    width = pad + cols * (thumb[0] + pad)
    height = pad + rows * (thumb[1] + label_h + pad)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/mnt/c/Windows/Fonts/arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for idx, row in enumerate(failures.itertuples(index=False)):
        col = idx % cols
        row_idx = idx // cols
        x = pad + col * (thumb[0] + pad)
        y = pad + row_idx * (thumb[1] + label_h + pad)
        with Image.open(row.path).convert("RGB") as image:
            image.thumbnail(thumb)
            tile = Image.new("RGB", thumb, "white")
            tile.paste(image, ((thumb[0] - image.width) // 2, (thumb[1] - image.height) // 2))
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb[0], y + thumb[1]), outline=(190, 190, 190), width=1)
        label = f"{row.user_species}: {row.pred_disease}\np={row.p_max:.3f}, H={row.entropy:.3f}"
        draw.text((x, y + thumb[1] + 5), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data/ood_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/agent/ood_domain_gate_mc30"))
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    images = find_images(args.image_dir, args.limit)
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import os

    os.environ["VETDERM_MC_PASSES"] = str(args.mc_passes)

    from vetderm_agent.config import get_settings
    from vetderm_agent.inference import VetDermInferenceService

    settings = get_settings()
    service = VetDermInferenceService(settings)
    service.load()

    print("VetDerm SGCA OOD evaluation")
    print(f"device: {service.device}")
    print(f"mc_passes: {args.mc_passes}")
    print(f"images: {len(images)}")
    print(f"species per image: {len(SPECIES)}")

    rows: list[dict[str, Any]] = []
    for image_index, image_path in enumerate(images, start=1):
        image_bytes = image_path.read_bytes()
        for species in SPECIES:
            result = service.predict_bytes(image_bytes, user_species=species, include_gradcam=False)
            unsupported = result.get("status") == "unsupported_image"
            top1 = (
                {"disease": "unsupported_image", "probability": 0.0}
                if unsupported
                else result["top3_diseases"][0]
            )
            signals = list(result["referral"]["signals"])
            flags = flag_dict(signals)
            rows.append(
                {
                    "image_id": image_path.stem,
                    "path": str(image_path),
                    "user_species": species,
                    "pred_species": result["model_species_prediction"],
                    "pred_species_confidence": float(result["model_species_confidence"]),
                    "pred_disease": top1["disease"],
                    "p_max": float(top1["probability"]),
                    "status": str(result.get("status", "ok")),
                    "domain_gate_score": result.get("domain_gate", {}).get("score"),
                    "domain_gate_threshold": result.get("domain_gate", {}).get("threshold"),
                    "domain_gate_reason": result.get("domain_gate", {}).get("reason"),
                    "domain_gate_nearest_species": result.get("domain_gate", {}).get("nearest_species"),
                    "domain_gate_nearest_disease": result.get("domain_gate", {}).get("nearest_disease"),
                    "entropy": float(result["uncertainty"]["predictive_entropy"] or 0.0),
                    "entropy_threshold": float(result["uncertainty"]["entropy_threshold"]),
                    "mutual_info": float(result["uncertainty"]["mutual_information"] or 0.0),
                    "conformal_size": int(len(result["conformal_prediction_set"])),
                    "conformal_set": "|".join(item["disease"] for item in result["conformal_prediction_set"]),
                    "referral": result["referral"]["level"],
                    "suppress_information_panel": bool(result["referral"]["suppress_information_panel"]),
                    "signals": "|".join(signals),
                    **flags,
                    "major_signal_count": major_signal_count(flags),
                    "safe_escalation": result["referral"]["level"] != "educational_only",
                    "top3_diseases": "|".join(item["disease"] for item in result["top3_diseases"]),
                    "top3_probabilities": "|".join(f"{item['probability']:.8f}" for item in result["top3_diseases"]),
                }
            )
        if image_index % 10 == 0 or image_index == len(images):
            print(f"processed {image_index}/{len(images)} images", flush=True)

    df = pd.DataFrame(rows)
    pred_path = args.output_dir / "ood_agent_predictions.csv"
    df.to_csv(pred_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summaries = write_summary_tables(df, args.output_dir)
    make_failure_contact_sheet(df, args.output_dir / "ood_unflagged_educational_contact_sheet.png")

    print(f"predictions: {pred_path}")
    print(f"summary: {args.output_dir / 'ood_referral_summary.csv'}")
    print("\nReferral summary")
    print(summaries["referral"].to_string(index=False))
    print("\nTop predicted labels")
    print(summaries["labels"].groupby("user_species", group_keys=False).head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
