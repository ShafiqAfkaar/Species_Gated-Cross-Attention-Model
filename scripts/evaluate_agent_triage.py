#!/usr/bin/env python3
"""Run full-cohort VetDerm SGCA triage-agent evaluation.

This script mirrors the deployed agent's inference and referral logic, but
keeps the evaluation batched and disables LLM/Grad-CAM/audit work. It writes
one per-image CSV per cohort plus summary tables and publication-ready plots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sgca.models import (  # noqa: E402
    CLASS_TO_SPECIES,
    build_dataloaders_from_split_dirs,
    build_evaluation_loader,
    index_species_disease_dataset,
    load_checkpoint,
    seed_everything,
)
from vetderm_agent.calibration import CalibrationProfile, entropy  # noqa: E402
from vetderm_agent.config import get_settings  # noqa: E402
from vetderm_agent.safety import evaluate_referral  # noqa: E402


SIGNAL_COLUMNS = {
    "flag_entropy": "high_predictive_entropy",
    "flag_conformal": "ambiguous_conformal_set",
    "flag_low_probability": "low_top1_probability",
    "flag_non_normal": "non_normal_triage_label",
    "flag_zoonotic": "zoonotic_or_contagious_possible",
    "flag_herd_health": "herd_health_or_reportable_possible",
}


def valid_disease_indices(species: str, disease_names: list[str]) -> np.ndarray:
    valid = [idx for idx, disease in enumerate(disease_names) if CLASS_TO_SPECIES.get(disease) == species]
    if not valid:
        raise ValueError(f"No valid disease labels for species={species!r}")
    return np.asarray(valid, dtype=np.int64)


def renormalize_to_indices(probs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    conditioned = np.zeros_like(probs, dtype=np.float64)
    total = float(np.asarray(probs, dtype=np.float64)[valid].sum())
    if total <= 0:
        conditioned[valid] = 1.0 / len(valid)
    else:
        conditioned[valid] = np.asarray(probs, dtype=np.float64)[valid] / total
    return conditioned


def image_quality_warning_from_path(path: str, *, min_side: int = 224) -> str | None:
    with Image.open(path) as img:
        width, height = img.size
    if min(width, height) < min_side:
        return f"image_short_side_below_{min_side}px"
    ratio = max(width, height) / max(1, min(width, height))
    if ratio > 3.0:
        return "extreme_aspect_ratio"
    return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def deterministic_probs(model: nn.Module, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    species_logits, disease_logits = model(images)
    species_probs = F.softmax(species_logits, dim=1).detach().cpu().numpy()
    disease_probs = F.softmax(disease_logits, dim=1).detach().cpu().numpy()
    return species_probs, disease_probs


@torch.no_grad()
def mc_disease_probs(model: nn.Module, images: torch.Tensor, *, passes: int) -> np.ndarray:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    all_passes = []
    for _ in range(passes):
        _, disease_logits = model(images)
        all_passes.append(F.softmax(disease_logits, dim=1).detach().cpu().numpy())
    model.eval()
    return np.stack(all_passes, axis=0)


def batch_conditioned_probs(
    all_passes: np.ndarray,
    user_species: list[str],
    disease_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean conditioned probabilities and per-pass conditioned probabilities."""
    conditioned_passes = []
    for pass_probs in all_passes:
        rows = []
        for probs, species in zip(pass_probs, user_species):
            rows.append(renormalize_to_indices(probs, valid_disease_indices(species, disease_names)))
        conditioned_passes.append(np.stack(rows, axis=0))
    conditioned_passes_np = np.stack(conditioned_passes, axis=0)
    return conditioned_passes_np.mean(axis=0), conditioned_passes_np


def topk_row(probs: np.ndarray, disease_names: list[str], k: int = 3) -> list[dict[str, Any]]:
    order = np.argsort(-probs)[:k]
    return [
        {"rank": rank + 1, "index": int(idx), "disease": disease_names[int(idx)], "probability": float(probs[int(idx)])}
        for rank, idx in enumerate(order)
    ]


def signal_flags(signals: list[str]) -> dict[str, bool]:
    flags = {column: signal in signals for column, signal in SIGNAL_COLUMNS.items()}
    flags["flag_image_quality"] = any(signal.startswith("image_quality:") for signal in signals)
    return flags


def major_signal_count(flags: dict[str, bool]) -> int:
    return int(
        flags["flag_entropy"]
        + flags["flag_conformal"]
        + flags["flag_low_probability"]
        + flags["flag_zoonotic"]
        + flags["flag_herd_health"]
        + flags["flag_image_quality"]
    )


def evaluate_loader(
    *,
    cohort: str,
    loader,
    model: nn.Module,
    device: torch.device,
    species_names: list[str],
    disease_names: list[str],
    calibration: CalibrationProfile,
    mc_passes: int,
    hash_images: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    n_seen = 0

    for images, species_labels, disease_labels, paths in loader:
        images = images.to(device, non_blocking=True)
        true_species = [species_names[int(idx)] for idx in species_labels.numpy()]
        true_diseases = [disease_names[int(idx)] for idx in disease_labels.numpy()]

        species_probs, raw_probs = deterministic_probs(model, images)
        all_passes = mc_disease_probs(model, images, passes=mc_passes)
        disease_probs, pass_conditioned = batch_conditioned_probs(all_passes, true_species, disease_names)

        pred_entropy = entropy(disease_probs)
        pass_entropy = entropy(pass_conditioned)
        mutual_info = np.maximum(pred_entropy - pass_entropy.mean(axis=0), 0.0)

        for i, path in enumerate(paths):
            user_species = true_species[i]
            gt_disease = true_diseases[i]
            valid = valid_disease_indices(user_species, disease_names)
            top3 = topk_row(disease_probs[i], disease_names, k=min(3, len(valid)))
            top1 = top3[0]
            conformal_indices = calibration.aps_set(disease_probs[i], valid)
            conformal_names = [disease_names[int(idx)] for idx in conformal_indices]
            image_quality_warning = image_quality_warning_from_path(path)

            referral = evaluate_referral(
                top1_disease=top1["disease"],
                top1_probability=top1["probability"],
                predictive_entropy=float(pred_entropy[i]),
                conformal_set_size=len(conformal_indices),
                entropy_threshold=calibration.entropy_threshold,
                probability_threshold=get_settings().probability_threshold,
                conformal_ambiguity_probability_threshold=get_settings().conformal_ambiguity_probability_threshold,
                suppress_after_signal_count=get_settings().suppress_info_after_signal_count,
                image_quality_warning=image_quality_warning,
            )
            flags = signal_flags(referral.signals)
            pred_species_idx = int(np.argmax(species_probs[i]))
            pred_disease = top1["disease"]
            rows.append(
                {
                    "cohort": cohort,
                    "image_hash": sha256_file(path) if hash_images else hashlib.sha256(str(path).encode()).hexdigest(),
                    "path": path,
                    "gt_species": user_species,
                    "user_species": user_species,
                    "gt_disease": gt_disease,
                    "pred_species": species_names[pred_species_idx],
                    "pred_species_confidence": float(np.max(species_probs[i])),
                    "pred_disease": pred_disease,
                    "p_max": float(top1["probability"]),
                    "raw_disease_top1_probability": float(np.max(raw_probs[i])),
                    "entropy": float(pred_entropy[i]),
                    "entropy_threshold": float(calibration.entropy_threshold),
                    "mutual_info": float(mutual_info[i]),
                    "conformal_size": int(len(conformal_indices)),
                    "conformal_set": "|".join(conformal_names),
                    "conformal_contains_gt": bool(gt_disease in conformal_names),
                    "image_quality_warning": image_quality_warning or "",
                    "signals": "|".join(referral.signals),
                    **flags,
                    "major_signal_count": major_signal_count(flags),
                    "referral": referral.level,
                    "suppress_information_panel": bool(referral.suppress_information_panel),
                    "correct": bool(pred_disease == gt_disease),
                    "top3_diseases": "|".join(item["disease"] for item in top3),
                    "top3_probabilities": "|".join(f"{item['probability']:.8f}" for item in top3),
                }
            )

        n_seen += len(paths)
        if n_seen and n_seen % 512 == 0:
            elapsed = time.perf_counter() - start
            print(f"{cohort}: processed {n_seen} images in {elapsed:.1f}s", flush=True)

    elapsed = time.perf_counter() - start
    print(f"{cohort}: completed {len(rows)} images in {elapsed:.1f}s", flush=True)
    return pd.DataFrame(rows)


def bootstrap_accuracy_ci(correct: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    if len(correct) == 0 or resamples <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    n = len(correct)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        values[i] = float(correct[idx].mean())
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def summarize_referral(df: pd.DataFrame, *, resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    for cohort, cohort_df in df.groupby("cohort", sort=False):
        for referral in ["educational_only", "vet_recommended", "vet_required", "all"]:
            part = cohort_df if referral == "all" else cohort_df[cohort_df["referral"] == referral]
            correct = part["correct"].to_numpy(dtype=bool)
            ci_low, ci_high = bootstrap_accuracy_ci(correct, resamples=resamples, seed=seed)
            labels = sorted(part["gt_disease"].unique()) if len(part) else []
            rows.append(
                {
                    "cohort": cohort,
                    "referral": referral,
                    "n": int(len(part)),
                    "fraction": float(len(part) / len(cohort_df)) if len(cohort_df) else np.nan,
                    "accuracy": float(correct.mean()) if len(part) else np.nan,
                    "accuracy_ci_low": ci_low,
                    "accuracy_ci_high": ci_high,
                    "macro_f1": float(
                        f1_score(
                            part["gt_disease"],
                            part["pred_disease"],
                            labels=labels,
                            average="macro",
                            zero_division=0,
                        )
                    )
                    if len(part)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_signals(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(SIGNAL_COLUMNS.keys()) + ["flag_image_quality"]
    rows = []
    for cohort, cohort_df in df.groupby("cohort", sort=False):
        for column in columns:
            rows.append(
                {
                    "cohort": cohort,
                    "signal": column,
                    "n": int(cohort_df[column].sum()),
                    "rate": float(cohort_df[column].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_decision_confusion(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, cohort_df in df.groupby("cohort", sort=False):
        educational = cohort_df["referral"] == "educational_only"
        correct = cohort_df["correct"].astype(bool)
        n = len(cohort_df)
        cells = {
            "true_clears_correct_educational": correct & educational,
            "safety_misses_incorrect_educational": (~correct) & educational,
            "wasted_referrals_correct_referred": correct & (~educational),
            "true_catches_incorrect_referred": (~correct) & (~educational),
        }
        row = {"cohort": cohort, "n": int(n)}
        for name, mask in cells.items():
            row[name] = int(mask.sum())
            row[f"{name}_rate"] = float(mask.sum() / n) if n else np.nan
        row["safety_miss_rate_within_educational"] = (
            float(((~correct) & educational).sum() / educational.sum()) if educational.sum() else 0.0
        )
        row["referred_fraction"] = float((~educational).mean()) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def risk_coverage_table(df: pd.DataFrame, coverages: list[float]) -> pd.DataFrame:
    rows = []
    for cohort, cohort_df in df.groupby("cohort", sort=False):
        ordered = cohort_df.sort_values("p_max", ascending=False).reset_index(drop=True)
        n = len(ordered)
        for coverage in coverages:
            k = max(1, int(np.ceil(n * coverage)))
            part = ordered.iloc[:k]
            rows.append(
                {
                    "cohort": cohort,
                    "coverage": coverage,
                    "n_retained": int(k),
                    "p_threshold_min_retained": float(part["p_max"].min()),
                    "accuracy": float(part["correct"].mean()),
                    "macro_f1": float(
                        f1_score(part["gt_disease"], part["pred_disease"], average="macro", zero_division=0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_triage_performance(df: pd.DataFrame, out_path: Path) -> None:
    cohorts = list(dict.fromkeys(df["cohort"]))
    referral_order = ["educational_only", "vet_recommended", "vet_required"]
    colors = {
        "educational_only": "#4C78A8",
        "vet_recommended": "#F58518",
        "vet_required": "#E45756",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=180)

    for cohort, ax in zip([c for c in cohorts if c != "val"][:2], axes[0]):
        part = df[df["cohort"] == cohort].sort_values("p_max", ascending=False).reset_index(drop=True)
        coverage = np.arange(1, len(part) + 1) / len(part)
        cumulative_acc = part["correct"].astype(float).expanding().mean()
        ax.plot(coverage, cumulative_acc, color="#1f77b4", linewidth=2)
        for ref in [0.7, 0.8, 0.9]:
            ax.axvline(ref, color="0.75", linestyle="--", linewidth=1)
        ax.set_title(f"Risk-coverage: {cohort}")
        ax.set_xlabel("Coverage retained")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)

    ax = axes[1, 0]
    width = 0.22
    x = np.arange(len(cohorts))
    for offset, referral in enumerate(referral_order):
        vals = []
        for cohort in cohorts:
            part = df[(df["cohort"] == cohort) & (df["referral"] == referral)]
            vals.append(float(part["correct"].mean()) if len(part) else np.nan)
        ax.bar(x + (offset - 1) * width, vals, width=width, label=referral, color=colors[referral])
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by referral level")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    bottom = np.zeros(len(cohorts))
    for referral in referral_order:
        vals = []
        for cohort in cohorts:
            cohort_df = df[df["cohort"] == cohort]
            vals.append(float((cohort_df["referral"] == referral).mean()))
        ax.bar(x, vals, bottom=bottom, label=referral, color=colors[referral])
        bottom += np.asarray(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, rotation=15, ha="right")
    ax.set_ylabel("Fraction of cohort")
    ax.set_title("Referral distribution")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_decision_confusion(df: pd.DataFrame, out_path: Path) -> None:
    cohorts = [c for c in ["internal_test", "development_external"] if c in set(df["cohort"])]
    if not cohorts:
        cohorts = list(dict.fromkeys(df["cohort"]))[:2]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(6 * len(cohorts), 4.8), dpi=180)
    if len(cohorts) == 1:
        axes = [axes]
    for ax, cohort in zip(axes, cohorts):
        part = df[df["cohort"] == cohort]
        educational = part["referral"] == "educational_only"
        correct = part["correct"].astype(bool)
        matrix = np.asarray(
            [
                [int((correct & educational).sum()), int((correct & (~educational)).sum())],
                [int(((~correct) & educational).sum()), int(((~correct) & (~educational)).sum())],
            ]
        )
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(cohort)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["educational", "referred"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["correct", "incorrect"])
        for (i, j), value in np.ndenumerate(matrix):
            color = "white" if value > matrix.max() * 0.55 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
        ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor="#E45756", linewidth=3))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(get_settings().checkpoint_path))
    parser.add_argument("--calibration-npz", default=str(get_settings().calibration_npz))
    parser.add_argument("--split-root", default="data/training_data_deduped_splits/seed42")
    parser.add_argument("--external-root", default="data/development_external_2026_05_02")
    parser.add_argument("--output-dir", default="results/agent/triage")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mc-passes", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-images", action="store_true", help="Use byte SHA-256 hashes instead of path hashes.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / f"mc{args.mc_passes}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(Path(args.checkpoint), pretrained=False, device=device)
    species_names = list(ckpt["species_names"])
    disease_names = list(ckpt["disease_names"])
    img_size = int(ckpt["img_size"])
    calibration = CalibrationProfile.from_npz(Path(args.calibration_npz))

    split_root = Path(args.split_root)
    data = build_dataloaders_from_split_dirs(
        split_root / "train",
        split_root / "val",
        split_root / "test",
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_policy="standard",
    )
    external_frame = index_species_disease_dataset(Path(args.external_root))
    external_loader, _ = build_evaluation_loader(
        external_frame,
        img_size=img_size,
        species_names=species_names,
        disease_names=disease_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loaders = {
        "val": data.val_dl,
        "internal_test": data.test_dl,
        "development_external": external_loader,
    }

    print(
        json.dumps(
            {
                "device": str(device),
                "checkpoint": str(args.checkpoint),
                "calibration_npz": str(args.calibration_npz),
                "calibration_source": calibration.source,
                "entropy_threshold": calibration.entropy_threshold,
                "aps_quantile": calibration.aps_quantile,
                "mc_passes": args.mc_passes,
                "output_dir": str(out_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    cohort_frames = []
    for cohort, loader in loaders.items():
        frame = evaluate_loader(
            cohort=cohort,
            loader=loader,
            model=model,
            device=device,
            species_names=species_names,
            disease_names=disease_names,
            calibration=calibration,
            mc_passes=args.mc_passes,
            hash_images=args.hash_images,
        )
        frame.to_csv(out_dir / f"{cohort}_agent_predictions.csv", index=False)
        cohort_frames.append(frame)

    all_df = pd.concat(cohort_frames, ignore_index=True)
    all_df.to_csv(out_dir / "all_agent_predictions.csv", index=False)

    referral = summarize_referral(all_df, resamples=args.resamples, seed=args.seed)
    signals = summarize_signals(all_df)
    decisions = summarize_decision_confusion(all_df)
    risk = risk_coverage_table(all_df, [1.0, 0.9, 0.8, 0.7])

    referral.to_csv(out_dir / "agent_referral_strata_summary.csv", index=False)
    signals.to_csv(out_dir / "agent_signal_firing_rates.csv", index=False)
    decisions.to_csv(out_dir / "agent_decision_confusion.csv", index=False)
    risk.to_csv(out_dir / "agent_risk_coverage_fixed_points.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    plot_triage_performance(all_df, out_dir / "fig_agent_triage_performance.png")
    plot_decision_confusion(all_df, out_dir / "fig_agent_decision_confusion.png")

    print("\nReferral strata summary:")
    print(referral.to_string(index=False))
    print("\nDecision confusion:")
    print(decisions.to_string(index=False))
    print("\nRisk coverage fixed points:")
    print(risk.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
