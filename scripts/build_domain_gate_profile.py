#!/usr/bin/env python3
"""Calibrate the VetDerm unsupported-image gate from SGCA feature vectors.

The gate stores a reference bank of L2-normalised fused SGCA vectors from the
training split. The decision threshold is the lower-tail quantile of validation
nearest-neighbour similarity to that reference bank, so the expected validation
false-rejection rate is controlled by --false-reject-rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sgca.models import build_eval_transform, load_checkpoint, seed_everything
from vetderm_agent.config import get_settings


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SpeciesDiseaseFolderDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.root = root
        self.transform = transform
        self.rows: list[tuple[Path, str, str]] = []
        for species_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for disease_dir in sorted(path for path in species_dir.iterdir() if path.is_dir()):
                for image_path in sorted(disease_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                        self.rows.append((image_path, species_dir.name, disease_dir.name))
        if not self.rows:
            raise FileNotFoundError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path, species, disease = self.rows[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), species, disease, str(path)


@torch.no_grad()
def extract_features(model, loader, device, species_names: list[str], disease_names: list[str]) -> dict[str, np.ndarray]:
    model.eval()
    features = []
    species = []
    diseases = []
    paths = []
    for batch_index, (images, species_batch, disease_batch, batch_paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        _, _, _, fused_vec, _ = model.forward_features(images, return_attention=False)
        fused_vec = F.normalize(fused_vec, dim=1)
        features.append(fused_vec.detach().cpu().numpy().astype(np.float32))
        species.extend(str(item) for item in species_batch)
        diseases.extend(str(item) for item in disease_batch)
        paths.extend(str(path) for path in batch_paths)
        if batch_index % 50 == 0:
            print(f"  extracted {batch_index * len(images):,} images", flush=True)
    return {
        "features": np.concatenate(features, axis=0),
        "species": np.asarray(species, dtype=object),
        "diseases": np.asarray(diseases, dtype=object),
        "paths": np.asarray(paths, dtype=object),
    }


def max_similarity(query: np.ndarray, reference: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
    out = np.empty(len(query), dtype=np.float32)
    ref_t = reference.T
    for start in range(0, len(query), chunk_size):
        sims = query[start : start + chunk_size] @ ref_t
        out[start : start + chunk_size] = sims.max(axis=1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=get_settings().checkpoint_path)
    parser.add_argument("--split-root", type=Path, default=Path("data/training_data_deduped_splits/seed42"))
    parser.add_argument("--output", type=Path, default=Path("vetderm_agent/domain_gate/domain_gate_profile.npz"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--false-reject-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, pretrained=False, device=device)
    species_names = list(ckpt["species_names"])
    disease_names = list(ckpt["disease_names"])
    img_size = int(ckpt["img_size"])

    transform = build_eval_transform(img_size)
    train_ds = SpeciesDiseaseFolderDataset(args.split_root / "train", transform)
    val_ds = SpeciesDiseaseFolderDataset(args.split_root / "val", transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(json.dumps({"device": str(device), "img_size": img_size, "checkpoint": str(args.checkpoint)}, indent=2))
    print("extracting training reference features...", flush=True)
    print(f"train images: {len(train_ds):,}; validation images: {len(val_ds):,}", flush=True)
    train = extract_features(model, train_dl, device, species_names, disease_names)
    print(f"train reference: {train['features'].shape}", flush=True)
    print("extracting validation features...", flush=True)
    val = extract_features(model, val_dl, device, species_names, disease_names)
    print(f"validation: {val['features'].shape}", flush=True)

    val_scores = max_similarity(val["features"], train["features"])
    threshold = float(np.quantile(val_scores, args.false_reject_rate))
    val_reject_rate = float((val_scores < threshold).mean())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        reference_features=train["features"].astype(np.float16),
        reference_species=train["species"],
        reference_diseases=train["diseases"],
        reference_paths=train["paths"],
        threshold=np.asarray(threshold, dtype=np.float32),
        false_reject_rate=np.asarray(args.false_reject_rate, dtype=np.float32),
        validation_scores=val_scores.astype(np.float32),
        validation_paths=val["paths"],
        validation_species=val["species"],
        validation_diseases=val["diseases"],
    )

    summary = {
        "output": str(args.output),
        "reference_n": int(len(train["features"])),
        "validation_n": int(len(val["features"])),
        "threshold": threshold,
        "target_false_reject_rate": args.false_reject_rate,
        "observed_validation_reject_rate": val_reject_rate,
        "validation_score_min": float(val_scores.min()),
        "validation_score_median": float(np.median(val_scores)),
        "validation_score_max": float(val_scores.max()),
    }
    (args.output.parent / "domain_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
