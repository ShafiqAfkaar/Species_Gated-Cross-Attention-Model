#!/usr/bin/env python3
"""Verify that the deployed VetDerm agent can generate Grad-CAM artifacts.

This checks the real agent inference path:
    VetDermInferenceService.predict_bytes(..., include_gradcam=True)

The full-cohort agent evaluation disables Grad-CAM for speed, but the API and
Streamlit paths enable it by default. This script runs representative images
from all three species and verifies that each Grad-CAM artifact exists and is
nonblank.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat


DEFAULT_SAMPLES = [
    (
        "Cat",
        "Ringworm in Cat",
        Path("data/training_data_deduped_splits/seed42/test/Cat/Ringworm in Cat/02f6a7313a3a__aug_ringworm_in_cat__0_5821.jpg"),
    ),
    (
        "Cattles",
        "Lumpy Skin",
        Path("data/training_data_deduped_splits/seed42/test/Cattles/Lumpy Skin/00aae99890b3__curated__archive__4___cows_datasets__lumpy__img_.jpg"),
    ),
    (
        "Dog",
        "Ringworm in Dog",
        Path("data/training_data_deduped_splits/seed42/test/Dog/Ringworm in Dog/00a0e9766750__curated__archive__train__ringworm__ringworm_116_.jpg"),
    ),
]


def image_stats(path: Path) -> dict[str, Any]:
    with Image.open(path).convert("RGB") as image:
        stat = ImageStat.Stat(image)
        return {
            "size": image.size,
            "bbox": image.getbbox(),
            "mean": tuple(round(v, 2) for v in stat.mean),
            "stddev": tuple(round(v, 2) for v in stat.stddev),
        }


def mean_abs_diff(path_a: Path, path_b: Path) -> float:
    with Image.open(path_a).convert("RGB") as a, Image.open(path_b).convert("RGB") as b:
        diff = ImageChops.difference(a, b)
        stat = ImageStat.Stat(diff)
        return float(sum(stat.mean) / len(stat.mean))


def validate_artifacts(artifacts: dict[str, str]) -> dict[str, Any]:
    if "error" in artifacts:
        return {"ok": False, "error": str(artifacts["error"])}
    required = ["input", "overlay", "lesion", "crop"]
    missing = [key for key in required if key not in artifacts or not Path(artifacts[key]).exists()]
    if missing:
        return {"ok": False, "error": f"missing artifacts: {missing}"}

    paths = {key: Path(artifacts[key]) for key in required}
    stats = {key: image_stats(path) for key, path in paths.items()}
    blank = [key for key, value in stats.items() if value["bbox"] is None]
    overlay_diff = mean_abs_diff(paths["input"], paths["overlay"])
    if blank:
        return {"ok": False, "error": f"blank artifacts: {blank}", "stats": stats}
    if overlay_diff <= 1.0:
        return {"ok": False, "error": f"overlay too similar to input: mean_abs_diff={overlay_diff:.3f}", "stats": stats}
    return {
        "ok": True,
        "overlay_mean_abs_diff": round(overlay_diff, 3),
        "stats": stats,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def make_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    good_rows = [row for row in rows if row.get("check", {}).get("ok")]
    if not good_rows:
        return

    columns = ["input", "overlay", "lesion", "crop"]
    thumb_size = (192, 192)
    label_h = 44
    title_h = 32
    pad = 12
    width = pad + len(columns) * (thumb_size[0] + pad)
    height = pad + len(good_rows) * (title_h + thumb_size[1] + label_h + pad)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/mnt/c/Windows/Fonts/arial.ttf", 16)
        small_font = ImageFont.truetype("/mnt/c/Windows/Fonts/arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    y = pad
    for row in good_rows:
        title = (
            f"{row['species']} | expected: {row['expected']} | predicted: "
            f"{row['predicted']} ({row['probability']:.4f})"
        )
        draw.text((pad, y), title, fill=(0, 0, 0), font=font)
        y += title_h
        for col_idx, column in enumerate(columns):
            x = pad + col_idx * (thumb_size[0] + pad)
            image_path = Path(row["check"]["paths"][column])
            with Image.open(image_path).convert("RGB") as image:
                image.thumbnail(thumb_size)
                tile = Image.new("RGB", thumb_size, "white")
                tile.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y, x + thumb_size[0], y + thumb_size[1]), outline=(190, 190, 190), width=1)
            draw.text((x, y + thumb_size[1] + 6), column, fill=(0, 0, 0), font=small_font)
        y += thumb_size[1] + label_h + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-passes", type=int, default=5, help="Use a small MC count for this Grad-CAM smoke check.")
    parser.add_argument(
        "--gradcam-dir",
        type=Path,
        default=Path("vetderm_agent/static/gradcam_check"),
        help="Directory where generated Grad-CAM files are written.",
    )
    args = parser.parse_args()

    os.environ["VETDERM_MC_PASSES"] = str(args.mc_passes)
    os.environ["VETDERM_GRADCAM_DIR"] = str(args.gradcam_dir)

    from vetderm_agent.config import get_settings
    from vetderm_agent.inference import VetDermInferenceService

    settings = get_settings()
    service = VetDermInferenceService(settings)
    service.load()

    print("VetDerm SGCA Grad-CAM check")
    print(f"device: {service.device}")
    print(f"mc_passes: {args.mc_passes}")
    print(f"gradcam_dir: {settings.gradcam_dir}")
    print(f"samples: {len(DEFAULT_SAMPLES)}")

    failures = 0
    contact_rows: list[dict[str, Any]] = []
    for index, (species, expected, image_path) in enumerate(DEFAULT_SAMPLES, start=1):
        if not image_path.exists():
            print(f"\n{index}. MISSING sample: {image_path}")
            failures += 1
            continue

        result = service.predict_bytes(image_path.read_bytes(), user_species=species, include_gradcam=True)
        top1 = result["top3_diseases"][0]
        artifacts = result.get("gradcam_artifacts") or {}
        check = validate_artifacts(artifacts)
        if not check["ok"]:
            failures += 1
        contact_rows.append(
            {
                "species": species,
                "expected": expected,
                "predicted": top1["disease"],
                "probability": float(top1["probability"]),
                "check": check,
            }
        )

        print("\n" + "=" * 100)
        print(f"{index}. species={species}")
        print(f"image={image_path}")
        print(f"expected_disease={expected}")
        print(f"identified_disease={top1['disease']} probability={top1['probability']:.4f}")
        print(f"model_species_prediction={result['model_species_prediction']}")
        print(f"referral={result['referral']['level']} signals={result['referral']['signals']}")
        print(f"gradcam_ok={check['ok']}")
        if not check["ok"]:
            print(f"gradcam_error={check.get('error')}")
        else:
            print(f"overlay_mean_abs_diff={check['overlay_mean_abs_diff']}")
            for key, path in check["paths"].items():
                stat = check["stats"][key]
                print(f"{key}: {path} size={stat['size']} stddev={stat['stddev']}")

    print("\n" + "=" * 100)
    contact_sheet = settings.gradcam_dir / "gradcam_check_contact_sheet.png"
    make_contact_sheet(contact_rows, contact_sheet)
    if contact_sheet.exists():
        print(f"contact_sheet: {contact_sheet}")
    print(f"summary: {len(DEFAULT_SAMPLES) - failures}/{len(DEFAULT_SAMPLES)} Grad-CAM checks passed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
