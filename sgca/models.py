import argparse
import copy
import io
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

import timm

try:
    import cv2
except ImportError:
    cv2 = None

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _random_jpeg_compress(image: Image.Image, quality_range: tuple[int, int] = (35, 92)) -> Image.Image:
    """Simulate internet/social-media JPEG recompression on a PIL image."""
    buffer = io.BytesIO()
    quality = random.randint(*quality_range)
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def _random_downsample_upsample(image: Image.Image, scale_range: tuple[float, float] = (0.55, 0.90)) -> Image.Image:
    """Simulate low-resolution uploads while preserving final crop size."""
    width, height = image.size
    scale = random.uniform(*scale_range)
    low_width = max(16, int(width * scale))
    low_height = max(16, int(height * scale))
    down = image.resize((low_width, low_height), resample=Image.BICUBIC)
    return down.resize((width, height), resample=Image.BICUBIC).convert("RGB")


CLASS_TO_SPECIES = {
    "Cat_normal": "Cat",
    "Ear Mites in Cat": "Cat",
    "Eye Infection in Cat": "Cat",
    "Ringworm in Cat": "Cat",
    "Skin Allergy in Cat": "Cat",
    "scabies cat": "Cat",
    "Foot and Mouth disease": "Cattles",
    "Lumpy Skin": "Cattles",
    "Normal Skin": "Cattles",
    "Ringworm(cow)": "Cattles",
    "papiloma": "Cattles",
    "scabies cattle": "Cattles",
    "Dog_normal": "Dog",
    "Eye Infection in Dog": "Dog",
    "Fungal Infection in Dog": "Dog",
    "Hot Spots in Dog": "Dog",
    "Mange in Dog": "Dog",
    "Skin Allergy in Dog": "Dog",
    "Tick Infestation in Dog": "Dog",
    "Ringworm in Dog": "Dog",
    "Dermatitis in Dog": "Dog",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class VetDermDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row.path).convert("RGB")
        image = self.transform(image)
        return image, int(row.species), int(row.disease), row.path


@dataclass
class DataBundle:
    train_dl: DataLoader
    val_dl: DataLoader
    test_dl: DataLoader
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    species_names: List[str]
    disease_names: List[str]
    img_size: int


def infer_species_from_class_name(class_name: str) -> str:
    species = CLASS_TO_SPECIES.get(class_name)
    if species is None:
        raise KeyError(f"Unknown class-to-species mapping for: {class_name}")
    return species


def index_species_disease_dataset(data_dir: Path) -> pd.DataFrame:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir.resolve()}")

    samples = []
    for species_dir in sorted([d for d in data_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
        for disease_dir in sorted([d for d in species_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
            for image_path in sorted(disease_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in valid_exts:
                    samples.append(
                        {
                            "path": str(image_path),
                            "species_name": species_dir.name,
                            "disease_name": disease_dir.name,
                            "class_key": f"{species_dir.name}::{disease_dir.name}",
                        }
                    )
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError(f"No valid images found in {data_dir}")
    return df


def index_flat_class_dataset(data_dir: Path) -> pd.DataFrame:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir.resolve()}")

    samples = []
    for class_dir in sorted([d for d in data_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
        species_name = infer_species_from_class_name(class_dir.name)
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in valid_exts:
                samples.append(
                    {
                        "path": str(image_path),
                        "species_name": species_name,
                        "disease_name": class_dir.name,
                        "class_key": f"{species_name}::{class_dir.name}",
                    }
                )
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError(f"No valid images found in {data_dir}")
    return df


def build_train_transform(img_size: int, augmentation_policy: str = "standard"):
    augmentation_policy = augmentation_policy.lower()
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if augmentation_policy == "standard":
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
                T.ToTensor(),
                normalize,
            ]
        )

    if augmentation_policy == "robust":
        return T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.72, 1.0), ratio=(0.85, 1.18)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(22),
                T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.04)], p=0.85),
                T.RandomAutocontrast(p=0.25),
                T.RandomAdjustSharpness(sharpness_factor=1.6, p=0.25),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.18),
                T.ToTensor(),
                normalize,
            ]
        )

    if augmentation_policy == "internet_robust":
        return T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.78, 1.0), ratio=(0.85, 1.18)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(18),
                T.RandomPerspective(distortion_scale=0.08, p=0.25),
                T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05)], p=0.85),
                T.RandomAutocontrast(p=0.25),
                T.RandomAdjustSharpness(sharpness_factor=1.7, p=0.25),
                T.RandomApply([T.Lambda(_random_jpeg_compress)], p=0.55),
                T.RandomApply([T.Lambda(_random_downsample_upsample)], p=0.35),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.4))], p=0.22),
                T.ToTensor(),
                normalize,
            ]
        )

    if augmentation_policy == "augmix":
        augmix = T.AugMix(severity=3, mixture_width=3, alpha=1.0)
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(18),
                T.RandomApply([T.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14, hue=0.03)], p=0.65),
                augmix,
                T.ToTensor(),
                normalize,
            ]
        )

    raise ValueError(
        f"Unknown augmentation_policy={augmentation_policy!r}. "
        "Use standard, robust, internet_robust, or augmix."
    )


def build_eval_transform(img_size: int):
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def encode_species_disease_frames(frames: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[str], List[str]]:
    species_names = sorted(set().union(*[set(frame["species_name"]) for frame in frames]))
    disease_names = sorted(set().union(*[set(frame["disease_name"]) for frame in frames]))
    sp2i = {name: idx for idx, name in enumerate(species_names)}
    dis2i = {name: idx for idx, name in enumerate(disease_names)}
    encoded = []
    for frame in frames:
        out = frame.copy()
        out["species"] = out["species_name"].map(sp2i)
        out["disease"] = out["disease_name"].map(dis2i)
        if out["species"].isna().any() or out["disease"].isna().any():
            raise RuntimeError("Failed to encode species/disease labels.")
        encoded.append(out)
    return encoded, species_names, disease_names


def make_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    counts = train_df["disease"].value_counts().to_dict()
    weights = train_df["disease"].map(lambda label: 1.0 / counts[int(label)]).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def class_balanced_weights(labels: pd.Series, n_classes: int, beta: float = 0.9999) -> torch.Tensor:
    counts = np.bincount(labels.astype(int).to_numpy(), minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    effective_num = 1.0 - np.power(beta, counts[present])
    weights[present] = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights[present] = weights[present] / np.maximum(weights[present].mean(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders_from_split_dirs(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    augmentation_policy: str = "standard",
    use_weighted_sampler: bool = False,
) -> DataBundle:
    train_df = index_species_disease_dataset(train_dir)
    val_df = index_species_disease_dataset(val_dir)
    test_df = index_species_disease_dataset(test_dir)
    (train_df, val_df, test_df), species_names, disease_names = encode_species_disease_frames([train_df, val_df, test_df])

    train_tf = build_train_transform(img_size, augmentation_policy=augmentation_policy)
    eval_tf = build_eval_transform(img_size)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    sampler = make_weighted_sampler(train_df) if use_weighted_sampler else None
    train_dl = DataLoader(
        VetDermDataset(train_df, train_tf),
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
    )
    val_dl = DataLoader(VetDermDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_dl = DataLoader(VetDermDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)

    return DataBundle(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        species_names=species_names,
        disease_names=disease_names,
        img_size=img_size,
    )


def build_evaluation_loader(
    frame: pd.DataFrame,
    img_size: int,
    species_names: List[str],
    disease_names: List[str],
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, pd.DataFrame]:
    (eval_df,), _, _ = encode_species_disease_frames([frame])
    sp2i = {name: idx for idx, name in enumerate(species_names)}
    dis2i = {name: idx for idx, name in enumerate(disease_names)}
    eval_df["species"] = eval_df["species_name"].map(sp2i)
    eval_df["disease"] = eval_df["disease_name"].map(dis2i)
    if eval_df["species"].isna().any() or eval_df["disease"].isna().any():
        missing_species = sorted(set(eval_df.loc[eval_df["species"].isna(), "species_name"]))
        missing_disease = sorted(set(eval_df.loc[eval_df["disease"].isna(), "disease_name"]))
        raise ValueError(json.dumps({"missing_species": missing_species, "missing_diseases": missing_disease}))
    loader_kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(VetDermDataset(eval_df, build_eval_transform(img_size)), batch_size=batch_size, shuffle=False, **loader_kwargs)
    return loader, eval_df


def build_dataloaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> DataBundle:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir.resolve()}")

    species_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    sp2i = {name: idx for idx, name in enumerate(species_names)}
    samples = []

    for species_name in species_names:
        species_dir = data_dir / species_name
        for disease_dir in sorted([d for d in species_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
            for image_path in sorted(disease_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in valid_exts:
                    continue
                samples.append(
                    {
                        "path": str(image_path),
                        "species_name": species_name,
                        "species": sp2i[species_name],
                        "disease_name": disease_dir.name,
                        "class_key": f"{species_name}::{disease_dir.name}",
                    }
                )

    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No valid images found.")

    disease_names = sorted(df["disease_name"].unique().tolist())
    dis2i = {name: idx for idx, name in enumerate(disease_names)}
    df["disease"] = df["disease_name"].map(dis2i)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["class_key"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["class_key"],
    )

    train_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_dl = DataLoader(VetDermDataset(train_df, train_tf), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(VetDermDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_dl = DataLoader(VetDermDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)

    return DataBundle(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        species_names=species_names,
        disease_names=disease_names,
        img_size=img_size,
    )


def build_dataloaders_from_pre_split_dirs(
    train_dir: Path,
    hard_validation_dir: Path,
    external_validation_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataBundle:
    train_df = index_species_disease_dataset(train_dir)
    hard_df = index_species_disease_dataset(hard_validation_dir)
    ext_df = index_species_disease_dataset(external_validation_dir)

    species_names = sorted(set(train_df["species_name"]).union(hard_df["species_name"]).union(ext_df["species_name"]))
    disease_names = sorted(set(train_df["disease_name"]).union(hard_df["disease_name"]).union(ext_df["disease_name"]))
    sp2i = {name: idx for idx, name in enumerate(species_names)}
    dis2i = {name: idx for idx, name in enumerate(disease_names)}

    for frame in (train_df, hard_df, ext_df):
        frame["species"] = frame["species_name"].map(sp2i)
        frame["disease"] = frame["disease_name"].map(dis2i)

    train_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_dl = DataLoader(VetDermDataset(train_df, train_tf), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(VetDermDataset(hard_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_dl = DataLoader(VetDermDataset(ext_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)

    return DataBundle(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        train_df=train_df,
        val_df=hard_df,
        test_df=ext_df,
        species_names=species_names,
        disease_names=disease_names,
        img_size=img_size,
    )


def build_dataloaders_from_train_roots(
    base_train_dir: Path,
    curated_train_dir: Path,
    hard_validation_dir: Path,
    external_validation_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataBundle:
    base_train_df = index_species_disease_dataset(base_train_dir)
    curated_train_df = index_flat_class_dataset(curated_train_dir)
    train_df = pd.concat([base_train_df, curated_train_df], ignore_index=True)
    hard_df = index_flat_class_dataset(hard_validation_dir)
    ext_df = index_flat_class_dataset(external_validation_dir)

    species_names = sorted(set(train_df["species_name"]).union(hard_df["species_name"]).union(ext_df["species_name"]))
    disease_names = sorted(set(train_df["disease_name"]).union(hard_df["disease_name"]).union(ext_df["disease_name"]))
    sp2i = {name: idx for idx, name in enumerate(species_names)}
    dis2i = {name: idx for idx, name in enumerate(disease_names)}

    for frame in (train_df, hard_df, ext_df):
        frame["species"] = frame["species_name"].map(sp2i)
        frame["disease"] = frame["disease_name"].map(dis2i)

    train_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_dl = DataLoader(VetDermDataset(train_df, train_tf), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(VetDermDataset(hard_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_dl = DataLoader(VetDermDataset(ext_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)

    return DataBundle(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        train_df=train_df,
        val_df=hard_df,
        test_df=ext_df,
        species_names=species_names,
        disease_names=disease_names,
        img_size=img_size,
    )


def build_dataloaders_from_training_and_evaluation_dirs(
    training_dir: Path,
    hard_validation_dir: Path,
    external_validation_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataBundle:
    train_df = index_species_disease_dataset(training_dir)
    hard_df = index_flat_class_dataset(hard_validation_dir)
    ext_df = index_flat_class_dataset(external_validation_dir)

    species_names = sorted(set(train_df["species_name"]).union(hard_df["species_name"]).union(ext_df["species_name"]))
    disease_names = sorted(set(train_df["disease_name"]).union(hard_df["disease_name"]).union(ext_df["disease_name"]))
    sp2i = {name: idx for idx, name in enumerate(species_names)}
    dis2i = {name: idx for idx, name in enumerate(disease_names)}

    for frame in (train_df, hard_df, ext_df):
        frame["species"] = frame["species_name"].map(sp2i)
        frame["disease"] = frame["disease_name"].map(dis2i)

    train_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_dl = DataLoader(VetDermDataset(train_df, train_tf), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(VetDermDataset(hard_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_dl = DataLoader(VetDermDataset(ext_df, eval_tf), batch_size=batch_size, shuffle=False, **loader_kwargs)

    return DataBundle(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        train_df=train_df,
        val_df=hard_df,
        test_df=ext_df,
        species_names=species_names,
        disease_names=disease_names,
        img_size=img_size,
    )


class MixStyle(nn.Module):
    """Feature-statistic mixing for domain generalization."""

    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0 or x.size(0) < 2 or random.random() > self.p:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True)
        sig = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt()
        x_normed = (x - mu) / sig
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0), 1, 1, 1)).to(x.device)
        perm = torch.randperm(x.size(0), device=x.device)
        mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)
        return x_normed * sig_mix + mu_mix


class EfficientNetV2SFeatureBackbone(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        mixstyle_p: float = 0.0,
        mixstyle_alpha: float = 0.3,
        mixstyle_layers: Tuple[str, ...] = ("blocks.0", "blocks.1", "blocks.2"),
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        self.channels = self.backbone.feature_info.channels()[-1]
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if mixstyle_p > 0 else None
        self.mixstyle_layers = tuple(mixstyle_layers)
        self._mixstyle_hooks = []
        if self.mixstyle is not None:
            modules = dict(self.backbone.named_modules())
            missing = [name for name in self.mixstyle_layers if name not in modules]
            if missing:
                raise KeyError(f"MixStyle layer(s) not found in EfficientNetV2-S backbone: {missing}")
            for name in self.mixstyle_layers:
                self._mixstyle_hooks.append(modules[name].register_forward_hook(self._mixstyle_hook))

    def _mixstyle_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output):
        if self.mixstyle is None or not isinstance(output, torch.Tensor):
            return output
        return self.mixstyle(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[-1]


class TimmGlobalFeatureBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.channels = getattr(self.backbone, "num_features")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Some timm models, especially ViTs, require a fixed input size.
        patch_embed = getattr(self.backbone, "patch_embed", None)
        target_img_size = getattr(patch_embed, "img_size", None)
        if target_img_size is not None:
            if isinstance(target_img_size, int):
                target_img_size = (target_img_size, target_img_size)
            h, w = x.shape[-2:]
            if (h, w) != tuple(target_img_size):
                x = F.interpolate(x, size=tuple(target_img_size), mode="bilinear", align_corners=False)
        return self.backbone(x)


class SpeciesConditionedSpatialTokens(nn.Module):
    def __init__(self, channels: int, species_dim: int = 256):
        super().__init__()
        self.spatial_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bias_proj = nn.Linear(species_dim, channels)
        self.scale_proj = nn.Linear(species_dim, channels)
        self.token_norm = nn.LayerNorm(channels)

    def forward(self, feat_map: torch.Tensor, species_emb: torch.Tensor) -> torch.Tensor:
        spatial_tokens = self.spatial_proj(feat_map).flatten(2).transpose(1, 2)
        token_bias = self.bias_proj(species_emb).unsqueeze(1)
        token_scale = torch.sigmoid(self.scale_proj(species_emb)).unsqueeze(1)
        species_tokens = spatial_tokens * token_scale + token_bias
        return self.token_norm(species_tokens)


class DSANDualAttentionBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 4, local_filters: int = 128):
        super().__init__()
        reduced = max(channels // reduction_ratio, 32)
        self.ca_fc1 = nn.Linear(channels, reduced)
        self.ca_fc2 = nn.Linear(reduced, channels)
        self.sa_conv = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.local_conv1 = nn.Conv2d(channels, local_filters, kernel_size=3, padding=1)
        self.local_conv2 = nn.Conv2d(local_filters, local_filters, kernel_size=3, padding=1)
        self.global_dense = nn.Linear(channels, 256)
        self.fused_bn = nn.BatchNorm1d(256 + local_filters)
        self.fused_dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        gap = x.mean(dim=(2, 3))
        ca = F.relu(self.ca_fc1(gap))
        ca = torch.sigmoid(self.ca_fc2(ca)).unsqueeze(-1).unsqueeze(-1)
        x_ca = x * ca

        sa = torch.sigmoid(self.sa_conv(x_ca))
        x_att = x_ca * sa

        local = F.relu(self.local_conv1(x_att))
        local = F.relu(self.local_conv2(local))
        local_gap = local.mean(dim=(2, 3))

        global_dense = F.relu(self.global_dense(gap))
        fused = torch.cat([global_dense, local_gap], dim=1)
        fused = self.fused_bn(fused)
        fused = self.fused_dropout(fused)
        return x_att, fused, local


class VDANLesionTokenBranch(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.roi_ca_fc1 = nn.Linear(channels, max(channels // 4, 64))
        self.roi_ca_fc2 = nn.Linear(max(channels // 4, 64), channels)
        self.roi_spatial = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.tokens_ln1 = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.tokens_ln2 = nn.LayerNorm(channels)
        self.ffn1 = nn.Linear(channels, channels * ff_mult)
        self.ffn2 = nn.Linear(channels * ff_mult, channels)
        self.tokens_ln3 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        gap = x.mean(dim=(2, 3))
        roi_ca = F.relu(self.roi_ca_fc1(gap))
        roi_ca = torch.sigmoid(self.roi_ca_fc2(roi_ca)).unsqueeze(-1).unsqueeze(-1)
        x_ca = x * roi_ca

        roi_spatial = torch.sigmoid(self.roi_spatial(x_ca))
        x_sa = x_ca * roi_spatial

        tokens = x_sa.flatten(2).transpose(1, 2)
        t = self.tokens_ln1(tokens)
        attn_out, _ = self.mha(t, t, t, need_weights=False)
        t = tokens + attn_out

        t_ffn = self.tokens_ln2(t)
        t_ffn = F.gelu(self.ffn1(t_ffn))
        t_ffn = self.ffn2(t_ffn)
        t = t + t_ffn

        t = self.tokens_ln3(t)
        pooled = self.dropout(t.mean(dim=1))
        return x_sa, pooled, roi_spatial.squeeze(1)


class SpeciesGatedAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(channels)
        self.gate_dense = nn.Linear(channels, channels)

    def forward(
        self,
        disease_tokens: torch.Tensor,
        species_tokens: torch.Tensor,
        species_vec: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.mha(
            query=self.query_norm(disease_tokens),
            key=self.context_norm(species_tokens),
            value=self.context_norm(species_tokens),
            need_weights=return_attention,
        )
        fused = self.out_norm(disease_tokens + attn_out)
        gate = torch.sigmoid(self.gate_dense(species_vec)).unsqueeze(1)
        return fused * gate, attn_weights


class SGCACrossAttentionModel(nn.Module):
    def __init__(
        self,
        n_species: int,
        n_diseases: int,
        pretrained: bool = True,
        num_heads: int = 4,
        mixstyle_p: float = 0.0,
        mixstyle_alpha: float = 0.3,
    ):
        super().__init__()
        self.backbone = EfficientNetV2SFeatureBackbone(
            pretrained=pretrained,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
        c = self.backbone.channels
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.disease_token_proj = nn.Linear(c, c)
        self.species_embedding = nn.Linear(c, 256)
        self.species_output = nn.Linear(256, n_species)
        self.species_context = nn.Linear(256, c)
        self.species_token_builder = SpeciesConditionedSpatialTokens(c)
        self.sgca_block = SpeciesGatedAttention(c, num_heads=num_heads)
        self.disease_fc1 = nn.Linear(c, 256)
        self.disease_dropout = nn.Dropout(0.2)
        self.disease_output = nn.Linear(256, n_diseases)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        feat = self.backbone(x)
        pooled = self.global_pool(feat).flatten(1)
        species_emb = F.relu(self.species_embedding(pooled))
        species_logits = self.species_output(species_emb)
        species_vec = F.relu(self.species_context(species_emb))
        disease_tokens = self.disease_token_proj(feat.flatten(2).transpose(1, 2))
        species_tokens = self.species_token_builder(feat, species_emb)
        fused_tokens, attn_weights = self.sgca_block(
            disease_tokens,
            species_tokens,
            species_vec,
            return_attention=return_attention,
        )
        fused_vec = fused_tokens.mean(dim=1)
        dx = F.relu(self.disease_fc1(fused_vec))
        dx = self.disease_dropout(dx)
        disease_logits = self.disease_output(dx)
        if return_attention:
            return species_logits, disease_logits, attn_weights
        return species_logits, disease_logits


class SGCACrossAttentionUncertaintyGradCAMModel(nn.Module):
    """Canonical SGCA cross-attention with deployable uncertainty/Grad-CAM hooks."""

    def __init__(
        self,
        n_species: int,
        n_diseases: int,
        pretrained: bool = True,
        num_heads: int = 4,
        mixstyle_p: float = 0.0,
        mixstyle_alpha: float = 0.3,
    ):
        super().__init__()
        self.backbone = EfficientNetV2SFeatureBackbone(
            pretrained=pretrained,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
        c = self.backbone.channels
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.disease_token_proj = nn.Linear(c, c)
        self.species_embedding = nn.Linear(c, 256)
        self.species_output = nn.Linear(256, n_species)
        self.species_context = nn.Linear(256, c)
        self.species_token_builder = SpeciesConditionedSpatialTokens(c)
        self.sgca_block = SpeciesGatedAttention(c, num_heads=num_heads)
        self.disease_fc1 = nn.Linear(c, 256)
        self.disease_dropout = nn.Dropout(0.3)
        self.disease_output = nn.Linear(256, n_diseases)

    def forward_features(self, x: torch.Tensor, return_attention: bool = False):
        feat = self.backbone(x)
        pooled = self.global_pool(feat).flatten(1)
        species_emb = F.relu(self.species_embedding(pooled))
        species_logits = self.species_output(species_emb)
        species_vec = F.relu(self.species_context(species_emb))
        disease_tokens = self.disease_token_proj(feat.flatten(2).transpose(1, 2))
        species_tokens = self.species_token_builder(feat, species_emb)
        fused_tokens, attn_weights = self.sgca_block(
            disease_tokens,
            species_tokens,
            species_vec,
            return_attention=return_attention,
        )
        fused_vec = fused_tokens.mean(dim=1)
        dx = F.relu(self.disease_fc1(fused_vec))
        dx = self.disease_dropout(dx)
        disease_logits = self.disease_output(dx)
        return species_logits, disease_logits, feat, fused_vec, attn_weights

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        species_logits, disease_logits, _, _, attn_weights = self.forward_features(x, return_attention=return_attention)
        if return_attention:
            return species_logits, disease_logits, attn_weights
        return species_logits, disease_logits


class UnifiedSGCAUncertaintyModel(nn.Module):
    def __init__(
        self,
        n_species: int,
        n_diseases: int,
        pretrained: bool = True,
        num_heads: int = 4,
        mixstyle_p: float = 0.0,
        mixstyle_alpha: float = 0.3,
    ):
        super().__init__()
        self.backbone = EfficientNetV2SFeatureBackbone(
            pretrained=pretrained,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
        c = self.backbone.channels
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.disease_token_proj = nn.Linear(c, c)
        self.species_embedding = nn.Sequential(
            nn.Linear(c, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
        )
        self.species_output = nn.Linear(256, n_species)
        self.species_context = nn.Linear(256, c)
        self.species_token_builder = SpeciesConditionedSpatialTokens(c, species_dim=256)
        self.sgca_block = SpeciesGatedAttention(c, num_heads=num_heads)
        self.disease_head = nn.Sequential(
            nn.Linear(c, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, n_diseases),
        )

    def forward_features(self, x: torch.Tensor, return_attention: bool = False):
        feat = self.backbone(x)
        pooled = self.global_pool(feat).flatten(1)
        species_emb = self.species_embedding(pooled)
        species_logits = self.species_output(species_emb)
        species_vec = F.relu(self.species_context(species_emb))
        disease_tokens = self.disease_token_proj(feat.flatten(2).transpose(1, 2))
        species_tokens = self.species_token_builder(feat, species_emb)
        fused_tokens, attn_weights = self.sgca_block(
            disease_tokens,
            species_tokens,
            species_vec,
            return_attention=return_attention,
        )
        fused_vec = fused_tokens.mean(dim=1)
        disease_logits = self.disease_head(fused_vec)
        return species_logits, disease_logits, feat, fused_vec, attn_weights

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        species_logits, disease_logits, _, _, attn_weights = self.forward_features(x, return_attention=return_attention)
        if return_attention:
            return species_logits, disease_logits, attn_weights
        return species_logits, disease_logits


class FullResearchSGCAModel(nn.Module):
    def __init__(self, n_species: int, n_diseases: int, pretrained: bool = True, num_heads: int = 4):
        super().__init__()
        self.backbone = EfficientNetV2SFeatureBackbone(pretrained=pretrained)
        c = self.backbone.channels
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dual_attention = DSANDualAttentionBlock(c)
        self.species_hidden = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
        )
        self.species_projection = nn.Linear(128, 256)
        self.species_output = nn.Linear(256, n_species)
        self.species_context = nn.Linear(256, c)

        self.lesion_branch = VDANLesionTokenBranch(c, num_heads=min(8, num_heads * 2))
        self.disease_token_proj = nn.Linear(c, c)
        self.species_token_builder = SpeciesConditionedSpatialTokens(c, species_dim=256)
        self.sgca_block = SpeciesGatedAttention(c, num_heads=num_heads)

        self.disease_head = nn.Sequential(
            nn.Linear((3 * c) + 256 + 128, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, n_diseases),
        )

    def forward_features(self, x: torch.Tensor, return_attention: bool = False):
        feat = self.backbone(x)
        att_feat, fused_ds, _ = self.dual_attention(feat)

        species_hidden = self.species_hidden(fused_ds)
        species_emb = F.relu(self.species_projection(species_hidden))
        species_logits = self.species_output(species_emb)
        species_vec = F.relu(self.species_context(species_emb))

        lesion_feat, lesion_vec, lesion_mask = self.lesion_branch(att_feat)
        disease_tokens = self.disease_token_proj(lesion_feat.flatten(2).transpose(1, 2))
        species_tokens = self.species_token_builder(att_feat, species_emb)
        fused_tokens, attn_weights = self.sgca_block(
            disease_tokens,
            species_tokens,
            species_vec,
            return_attention=return_attention,
        )
        fused_vec = fused_tokens.mean(dim=1)
        pooled_feat = self.global_pool(att_feat).flatten(1)
        disease_input = torch.cat([fused_vec, lesion_vec, fused_ds, pooled_feat], dim=1)
        disease_logits = self.disease_head(disease_input)
        aux = {
            "feature_map": lesion_feat,
            "embedding": fused_vec,
            "lesion_vec": lesion_vec,
            "lesion_mask": lesion_mask,
        }
        return species_logits, disease_logits, aux, attn_weights

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        species_logits, disease_logits, aux, attn_weights = self.forward_features(
            x, return_attention=return_attention
        )
        if return_attention:
            return species_logits, disease_logits, aux, attn_weights
        return species_logits, disease_logits


class BaselineNoConditioning(nn.Module):
    def __init__(self, n_species: int, n_diseases: int, pretrained: bool = True):
        super().__init__()
        self.backbone = EfficientNetV2SFeatureBackbone(pretrained=pretrained)
        c = self.backbone.channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.species_fc = nn.Linear(c, 256)
        self.species_output = nn.Linear(256, n_species)
        self.disease_fc = nn.Linear(c, 256)
        self.disease_dropout = nn.Dropout(0.2)
        self.disease_output = nn.Linear(256, n_diseases)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        sp = F.relu(self.species_fc(pooled))
        species_logits = self.species_output(sp)
        dx = F.relu(self.disease_fc(pooled))
        dx = self.disease_dropout(dx)
        disease_logits = self.disease_output(dx)
        return species_logits, disease_logits


class BaselineConcatConditioning(nn.Module):
    def __init__(self, n_species: int, n_diseases: int, pretrained: bool = True):
        super().__init__()
        self.backbone = EfficientNetV2SFeatureBackbone(pretrained=pretrained)
        c = self.backbone.channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.species_fc = nn.Linear(c, 256)
        self.species_output = nn.Linear(256, n_species)
        self.disease_fc = nn.Linear(c + 256, 256)
        self.disease_dropout = nn.Dropout(0.2)
        self.disease_output = nn.Linear(256, n_diseases)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        sp = F.relu(self.species_fc(pooled))
        species_logits = self.species_output(sp)
        concat = torch.cat([pooled, sp], dim=1)
        dx = F.relu(self.disease_fc(concat))
        dx = self.disease_dropout(dx)
        disease_logits = self.disease_output(dx)
        return species_logits, disease_logits


class AblationNoGate(nn.Module):
    def __init__(
        self,
        n_species: int,
        n_diseases: int,
        pretrained: bool = True,
        num_heads: int = 4,
        mixstyle_p: float = 0.0,
        mixstyle_alpha: float = 0.3,
    ):
        super().__init__()
        self.backbone = EfficientNetV2SFeatureBackbone(
            pretrained=pretrained,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
        c = self.backbone.channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.species_embedding = nn.Linear(c, 256)
        self.species_output = nn.Linear(256, n_species)
        self.species_token_builder = SpeciesConditionedSpatialTokens(c)
        self.disease_token_proj = nn.Linear(c, c)
        self.query_norm = nn.LayerNorm(c)
        self.context_norm = nn.LayerNorm(c)
        self.mha = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(c)
        self.disease_fc1 = nn.Linear(c, 256)
        self.disease_dropout = nn.Dropout(0.2)
        self.disease_output = nn.Linear(256, n_diseases)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        species_emb = F.relu(self.species_embedding(pooled))
        species_logits = self.species_output(species_emb)
        disease_tokens = self.disease_token_proj(feat.flatten(2).transpose(1, 2))
        species_tokens = self.species_token_builder(feat, species_emb)
        attn_out, _ = self.mha(
            query=self.query_norm(disease_tokens),
            key=self.context_norm(species_tokens),
            value=self.context_norm(species_tokens),
            need_weights=False,
        )
        fused_tokens = self.out_norm(disease_tokens + attn_out)
        fused_vec = fused_tokens.mean(dim=1)
        dx = F.relu(self.disease_fc1(fused_vec))
        dx = self.disease_dropout(dx)
        disease_logits = self.disease_output(dx)
        return species_logits, disease_logits


class StandardTimmBaseline(nn.Module):
    def __init__(self, backbone_name: str, n_species: int, n_diseases: int, pretrained: bool = True):
        super().__init__()
        self.backbone = TimmGlobalFeatureBackbone(backbone_name, pretrained=pretrained)
        c = self.backbone.channels
        self.species_head = nn.Sequential(
            nn.Linear(c, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_species),
        )
        self.disease_head = nn.Sequential(
            nn.Linear(c, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_diseases),
        )

    def forward(self, x: torch.Tensor):
        pooled = self.backbone(x)
        species_logits = self.species_head(pooled)
        disease_logits = self.disease_head(pooled)
        return species_logits, disease_logits


class ResNet50Baseline(StandardTimmBaseline):
    def __init__(self, n_species: int, n_diseases: int, pretrained: bool = True):
        super().__init__("resnet50", n_species, n_diseases, pretrained=pretrained)


class EfficientNetV2SBaseline(StandardTimmBaseline):
    def __init__(self, n_species: int, n_diseases: int, pretrained: bool = True):
        super().__init__("tf_efficientnetv2_s", n_species, n_diseases, pretrained=pretrained)


class ViTB16Baseline(StandardTimmBaseline):
    def __init__(self, n_species: int, n_diseases: int, pretrained: bool = True):
        super().__init__("vit_base_patch16_224", n_species, n_diseases, pretrained=pretrained)


MODEL_REGISTRY = {
    "vetderm_hat": FullResearchSGCAModel,
    "unified_sgca_uncertainty": UnifiedSGCAUncertaintyModel,
    "sgca_cross_attention": SGCACrossAttentionModel,
    "sgca_cross_attention_uncertainty_gradcam": SGCACrossAttentionUncertaintyGradCAMModel,
    "baseline_no_conditioning": BaselineNoConditioning,
    "baseline_concat_conditioning": BaselineConcatConditioning,
    "ablation_no_gate": AblationNoGate,
    "resnet50_baseline": ResNet50Baseline,
    "efficientnetv2s_baseline": EfficientNetV2SBaseline,
    "vit_b16_baseline": ViTB16Baseline,
}

MODEL_DISPLAY_NAMES = {
    "vetderm_hat": "VetDerm-HAT",
    "unified_sgca_uncertainty": "Unified_SGCA_Uncertainty_GradCAM",
    "sgca_cross_attention": "SGCA_CrossAttention",
    "sgca_cross_attention_uncertainty_gradcam": "SGCA_CrossAttention_Uncertainty_GradCAM",
    "baseline_no_conditioning": "Baseline_NoConditioning",
    "baseline_concat_conditioning": "Baseline_ConcatConditioning",
    "ablation_no_gate": "Ablation_NoGate",
    "resnet50_baseline": "ResNet50_Baseline",
    "efficientnetv2s_baseline": "EfficientNetV2S_Baseline",
    "vit_b16_baseline": "ViT_B16_Baseline",
}


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(
        self,
        species_weight: Optional[torch.Tensor] = None,
        disease_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.log_s_sp = nn.Parameter(torch.zeros(1))
        self.log_s_dis = nn.Parameter(torch.zeros(1))
        self.label_smoothing = float(label_smoothing)
        self.register_buffer(
            "species_weight",
            species_weight.detach().float().clone() if species_weight is not None else torch.empty(0),
        )
        self.register_buffer(
            "disease_weight",
            disease_weight.detach().float().clone() if disease_weight is not None else torch.empty(0),
        )

    def forward(self, sp_logits, dis_logits, sp_labels, dis_labels):
        species_weight = self.species_weight if self.species_weight.numel() else None
        disease_weight = self.disease_weight if self.disease_weight.numel() else None
        loss_sp = F.cross_entropy(
            sp_logits,
            sp_labels,
            weight=species_weight,
            label_smoothing=self.label_smoothing,
        )
        loss_dis = F.cross_entropy(
            dis_logits,
            dis_labels,
            weight=disease_weight,
            label_smoothing=self.label_smoothing,
        )
        loss = (
            torch.exp(-self.log_s_sp) * loss_sp
            + self.log_s_sp
            + torch.exp(-self.log_s_dis) * loss_dis
            + self.log_s_dis
        )
        return loss, loss_sp.detach(), loss_dis.detach()


class FullResearchLoss(nn.Module):
    def __init__(
        self,
        n_species: int,
        n_diseases: int,
        gamma: float = 2.0,
        epsilon: float = 0.1,
        proto_lam: float = 0.1,
    ):
        super().__init__()
        self.n_species = n_species
        self.n_diseases = n_diseases
        self.gamma = gamma
        self.epsilon = epsilon
        self.proto_lam = proto_lam
        self.ce = nn.CrossEntropyLoss()
        self.log_s_sp = nn.Parameter(torch.zeros(1))
        self.log_s_dis = nn.Parameter(torch.zeros(1))

    def focal_ls(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        y_oh = F.one_hot(targets, self.n_diseases).float()
        q = (1 - self.epsilon) * y_oh + self.epsilon / self.n_diseases
        p_true = (probs * y_oh).sum(dim=1)
        focal_w = (1.0 - p_true).pow(self.gamma)
        log_p = torch.log(probs + 1e-8)
        ls_ce = -(q * log_p).sum(dim=1)
        return (focal_w * ls_ce).mean()

    def proto(self, emb: torch.Tensor, y_sp: torch.Tensor, y_dis: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        pulls = []
        protos = []
        proto_labels = []
        for sp in range(self.n_species):
            for dis in range(self.n_diseases):
                mask = (y_sp == sp) & (y_dis == dis)
                if mask.sum() < 2:
                    continue
                z = emb[mask]
                proto = z.mean(dim=0, keepdim=True)
                pulls.append(((z - proto) ** 2).sum(dim=1).mean())
                protos.append(proto.squeeze(0))
                proto_labels.append((sp, dis))

        if not protos:
            return emb.new_tensor(0.0)

        pulls_term = torch.stack(pulls).mean() if pulls else emb.new_tensor(0.0)
        pushes = []
        for i in range(len(protos)):
            for j in range(i + 1, len(protos)):
                if proto_labels[i] == proto_labels[j]:
                    continue
                d2 = ((protos[i] - protos[j]) ** 2).sum()
                pushes.append(F.relu(margin - d2))
        pushes_term = torch.stack(pushes).mean() if pushes else emb.new_tensor(0.0)
        return pulls_term + pushes_term

    def forward(
        self,
        sp_logits: torch.Tensor,
        dis_logits: torch.Tensor,
        sp_labels: torch.Tensor,
        dis_labels: torch.Tensor,
        embedding: torch.Tensor,
    ):
        loss_sp = self.ce(sp_logits, sp_labels)
        loss_dis = self.focal_ls(dis_logits, dis_labels)
        loss_proto = self.proto(embedding, sp_labels, dis_labels)
        loss = (
            torch.exp(-self.log_s_sp) * loss_sp
            + self.log_s_sp
            + torch.exp(-self.log_s_dis) * (loss_dis + self.proto_lam * loss_proto)
            + self.log_s_dis
        )
        return loss, loss_sp.detach(), loss_dis.detach(), loss_proto.detach()


def pcgrad_step(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    shared_params: List[torch.nn.Parameter],
    loss_sp: torch.Tensor,
    loss_dis: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss_sp.backward(retain_graph=True)
    grads_sp = [None if p.grad is None else p.grad.detach().clone() for p in shared_params]

    optimizer.zero_grad(set_to_none=True)
    loss_dis.backward(retain_graph=True)
    grads_dis = [None if p.grad is None else p.grad.detach().clone() for p in shared_params]

    paired = [(g_sp, g_dis) for g_sp, g_dis in zip(grads_sp, grads_dis) if g_sp is not None and g_dis is not None]
    if paired:
        vec_sp = torch.cat([g_sp.flatten() for g_sp, _ in paired])
        vec_dis = torch.cat([g_dis.flatten() for _, g_dis in paired])
        cos_sim = float(F.cosine_similarity(vec_sp, vec_dis, dim=0).item())
    else:
        vec_sp = None
        vec_dis = None
        cos_sim = 0.0

    proj_grads_sp = grads_sp
    proj_grads_dis = grads_dis
    if cos_sim < 0.0 and vec_sp is not None and vec_dis is not None:
        dot = torch.dot(vec_sp, vec_dis)
        dis_norm = torch.dot(vec_dis, vec_dis).clamp_min(1e-12)
        sp_norm = torch.dot(vec_sp, vec_sp).clamp_min(1e-12)
        proj_grads_sp = []
        proj_grads_dis = []
        for g_sp, g_dis in zip(grads_sp, grads_dis):
            if g_sp is None or g_dis is None:
                proj_grads_sp.append(g_sp)
                proj_grads_dis.append(g_dis)
                continue
            proj_grads_sp.append(g_sp - (dot / dis_norm) * g_dis)
            proj_grads_dis.append(g_dis - (dot / sp_norm) * g_sp)

    total_loss = loss_sp + loss_dis
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    for p, g_sp, g_dis in zip(shared_params, proj_grads_sp, proj_grads_dis):
        if g_sp is None and g_dis is None:
            continue
        grad = 0
        if g_sp is not None:
            grad = grad + g_sp
        if g_dis is not None:
            grad = grad + g_dis
        p.grad = grad
    return cos_sim


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    all_sp_preds, all_sp_labels = [], []
    all_dis_preds, all_dis_labels = [], []
    paths = []

    for images, species_labels, disease_labels, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        species_labels = species_labels.to(device, non_blocking=True)
        disease_labels = disease_labels.to(device, non_blocking=True)
        species_logits, disease_logits = model(images)
        all_sp_preds.append(species_logits.argmax(dim=1).cpu().numpy())
        all_sp_labels.append(species_labels.cpu().numpy())
        all_dis_preds.append(disease_logits.argmax(dim=1).cpu().numpy())
        all_dis_labels.append(disease_labels.cpu().numpy())
        paths.extend(batch_paths)

    y_sp = np.concatenate(all_sp_labels)
    p_sp = np.concatenate(all_sp_preds)
    y_dis = np.concatenate(all_dis_labels)
    p_dis = np.concatenate(all_dis_preds)
    return {
        "species_acc": accuracy_score(y_sp, p_sp),
        "species_f1": f1_score(y_sp, p_sp, average="macro", zero_division=0),
        "disease_acc": accuracy_score(y_dis, p_dis),
        "disease_f1": f1_score(y_dis, p_dis, average="macro", zero_division=0),
        "y_sp": y_sp,
        "p_sp": p_sp,
        "y_dis": y_dis,
        "p_dis": p_dis,
        "paths": paths,
    }


def train_model(
    model_key: str,
    data: DataBundle,
    device: torch.device,
    results_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    pretrained: bool,
    use_amp: bool,
    class_balanced_loss: bool = False,
    cb_beta: float = 0.9999,
    label_smoothing: float = 0.0,
    mixstyle_p: float = 0.0,
    mixstyle_alpha: float = 0.3,
    early_stopping_patience: int = 4,
) -> Tuple[nn.Module, pd.DataFrame, Path]:
    model_kwargs = {}
    if model_key in {
        "unified_sgca_uncertainty",
        "sgca_cross_attention",
        "sgca_cross_attention_uncertainty_gradcam",
        "ablation_no_gate",
    }:
        model_kwargs = {"mixstyle_p": mixstyle_p, "mixstyle_alpha": mixstyle_alpha}
    model = MODEL_REGISTRY[model_key](
        len(data.species_names),
        len(data.disease_names),
        pretrained=pretrained,
        **model_kwargs,
    ).to(device)
    model_name = MODEL_DISPLAY_NAMES[model_key]
    disease_weight = (
        class_balanced_weights(data.train_df["disease"], len(data.disease_names), beta=cb_beta).to(device)
        if class_balanced_loss
        else None
    )
    species_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    disease_ce = nn.CrossEntropyLoss(weight=disease_weight, label_smoothing=label_smoothing)
    uncertainty_loss = (
        MultiTaskUncertaintyLoss(disease_weight=disease_weight, label_smoothing=label_smoothing).to(device)
        if model_key in {"unified_sgca_uncertainty", "sgca_cross_attention_uncertainty_gradcam"}
        else None
    )
    full_research_loss = (
        FullResearchLoss(len(data.species_names), len(data.disease_names)).to(device)
        if model_key == "vetderm_hat"
        else None
    )

    params = list(model.parameters())
    if uncertainty_loss is not None:
        params += list(uncertainty_loss.parameters())
    if full_research_loss is not None:
        params += list(full_research_loss.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = None
    best_aux_state = None
    best_val_f1 = -1.0
    patience = 0
    history = []
    best_path = results_dir / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_sp = 0.0
        running_dis = 0.0
        running_proto = 0.0
        pcgrad_cos = []

        for images, species_labels, disease_labels, _ in data.train_dl:
            images = images.to(device, non_blocking=True)
            species_labels = species_labels.to(device, non_blocking=True)
            disease_labels = disease_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_ctx:
                if full_research_loss is not None:
                    species_logits, disease_logits, aux, _ = model.forward_features(images, return_attention=False)
                    loss, loss_sp, loss_dis, loss_proto = full_research_loss(
                        species_logits,
                        disease_logits,
                        species_labels,
                        disease_labels,
                        aux["embedding"],
                    )
                else:
                    species_logits, disease_logits = model(images)
                if uncertainty_loss is not None:
                    loss, loss_sp, loss_dis = uncertainty_loss(
                        species_logits, disease_logits, species_labels, disease_labels
                    )
                    loss_proto = torch.zeros(1, device=device)
                elif full_research_loss is None:
                    loss_sp = species_ce(species_logits, species_labels)
                    loss_dis = disease_ce(disease_logits, disease_labels)
                    loss = loss_sp + 2.0 * loss_dis
                    loss_proto = torch.zeros(1, device=device)

            if full_research_loss is not None:
                with amp_ctx:
                    sp_logits_pg, dis_logits_pg, aux_pg, _ = model.forward_features(images, return_attention=False)
                    loss_sp_pg = full_research_loss.ce(sp_logits_pg, species_labels)
                    loss_dis_pg = full_research_loss.focal_ls(dis_logits_pg, disease_labels) + (
                        full_research_loss.proto_lam * full_research_loss.proto(
                            aux_pg["embedding"], species_labels, disease_labels
                        )
                    )
                shared_params = list(model.backbone.parameters()) + list(model.dual_attention.parameters()) + list(
                    model.sgca_block.parameters()
                )
                cos_sim = pcgrad_step(optimizer, model, shared_params, loss_sp_pg, loss_dis_pg)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pcgrad_cos.append(cos_sim)
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            running_loss += float(loss.item())
            running_sp += float(loss_sp.item())
            running_dis += float(loss_dis.item())
            running_proto += float(loss_proto.item())

        val_metrics = evaluate(model, data.val_dl, device)
        scheduler.step(val_metrics["disease_f1"])
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(data.train_dl), 1),
            "species_loss": running_sp / max(len(data.train_dl), 1),
            "disease_loss": running_dis / max(len(data.train_dl), 1),
            "val_species_acc": val_metrics["species_acc"],
            "val_species_f1": val_metrics["species_f1"],
            "val_disease_acc": val_metrics["disease_acc"],
            "val_disease_f1": val_metrics["disease_f1"],
        }
        if uncertainty_loss is not None:
            row["sigma_species"] = float(torch.exp(uncertainty_loss.log_s_sp / 2).item())
            row["sigma_disease"] = float(torch.exp(uncertainty_loss.log_s_dis / 2).item())
        if full_research_loss is not None:
            row["sigma_species"] = float(torch.exp(full_research_loss.log_s_sp / 2).item())
            row["sigma_disease"] = float(torch.exp(full_research_loss.log_s_dis / 2).item())
            row["proto_loss"] = running_proto / max(len(data.train_dl), 1)
            row["pcgrad_cosine"] = float(np.mean(pcgrad_cos)) if pcgrad_cos else 0.0
        history.append(row)

        print(
            f"Ep {epoch:02d} | loss={row['train_loss']:.4f} | "
            f"val_species_f1={row['val_species_f1']:.4f} | val_disease_f1={row['val_disease_f1']:.4f}"
        )

        if row["val_disease_f1"] > best_val_f1:
            best_val_f1 = row["val_disease_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_aux_state = copy.deepcopy(uncertainty_loss.state_dict()) if uncertainty_loss is not None else None
            patience = 0
            payload = {
                "model_key": model_key,
                "model_name": model_name,
                "state_dict": best_state,
                "species_names": data.species_names,
                "disease_names": data.disease_names,
                "img_size": data.img_size,
                "training_options": {
                    "class_balanced_loss": class_balanced_loss,
                    "cb_beta": cb_beta,
                    "label_smoothing": label_smoothing,
                    "mixstyle_p": mixstyle_p,
                    "mixstyle_alpha": mixstyle_alpha,
                },
            }
            if best_aux_state is not None:
                payload["uncertainty_state_dict"] = best_aux_state
            if full_research_loss is not None:
                payload["full_research_loss_state_dict"] = copy.deepcopy(full_research_loss.state_dict())
            torch.save(payload, best_path)
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / f"{model_name}_history.csv", index=False)
    return model, history_df, best_path


def mc_dropout_predict(model: nn.Module, loader: DataLoader, device: torch.device, passes: int = 20):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    all_passes = []
    labels = []
    paths = []
    with torch.no_grad():
        for t in range(passes):
            preds_t = []
            batch_labels = []
            batch_paths = []
            for images, _, disease_labels, item_paths in loader:
                images = images.to(device)
                _, disease_logits = model(images)
                preds_t.append(F.softmax(disease_logits, dim=1).cpu().numpy())
                batch_labels.append(disease_labels.numpy())
                batch_paths.extend(item_paths)
            all_passes.append(np.concatenate(preds_t, axis=0))
            if t == 0:
                labels = np.concatenate(batch_labels, axis=0)
                paths = batch_paths
    all_passes = np.array(all_passes)
    probs = all_passes.mean(axis=0)
    eps = 1e-8
    pred_entropy = -(probs * np.log(probs + eps)).sum(axis=1)
    pass_entropy = -(all_passes * np.log(all_passes + eps)).sum(axis=2)
    mutual_info = np.maximum(pred_entropy - pass_entropy.mean(axis=0), 0)
    return {
        "probs": probs,
        "preds": probs.argmax(axis=1),
        "labels": labels,
        "paths": paths,
        "pred_entropy": pred_entropy,
        "mutual_info": mutual_info,
    }


@torch.no_grad()
def get_probabilities(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs = []
    all_labels = []
    all_paths = []
    for images, _, disease_labels, item_paths in loader:
        images = images.to(device)
        _, disease_logits = model(images)
        all_probs.append(F.softmax(disease_logits, dim=1).cpu().numpy())
        all_labels.append(disease_labels.numpy())
        all_paths.extend(item_paths)
    return np.concatenate(all_probs), np.concatenate(all_labels), all_paths


def conformal_prediction_sets(
    probs_cal: np.ndarray,
    labels_cal: np.ndarray,
    probs_test: np.ndarray,
    labels_test: Optional[np.ndarray] = None,
    alpha: float = 0.1,
):
    n = len(labels_cal)
    scores = 1.0 - probs_cal[np.arange(n), labels_cal]
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(scores, level))
    thr = 1.0 - q_hat
    sets = []
    for p in probs_test:
        pred_set = list(np.where(p >= thr)[0])
        sets.append(pred_set if pred_set else [int(p.argmax())])
    out = {
        "alpha": alpha,
        "threshold": thr,
        "sets": sets,
        "mean_set_size": float(np.mean([len(s) for s in sets])),
        "singleton_rate": float(np.mean([len(s) == 1 for s in sets])),
    }
    if labels_test is not None:
        out["coverage"] = float(np.mean([labels_test[i] in sets[i] for i in range(len(labels_test))]))
    return out


class GradCAMExporter:
    def __init__(self, model: UnifiedSGCAUncertaintyModel):
        self.model = model

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)
        if hasattr(self.model, "forward_features"):
            forward_out = self.model.forward_features(input_tensor, return_attention=False)
            if isinstance(forward_out, tuple) and len(forward_out) >= 3:
                species_logits = forward_out[0]
                disease_logits = forward_out[1]
                fmap = forward_out[2]
            else:
                raise RuntimeError("forward_features did not return the expected feature map tensor.")
        else:
            raise RuntimeError("GradCAMExporter requires a model with forward_features().")
        if class_idx is None:
            class_idx = int(disease_logits.argmax(dim=1).item())
        score = disease_logits[:, class_idx].sum()
        grad = torch.autograd.grad(score, fmap, retain_graph=False, create_graph=False)[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * fmap).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    heat_uint8 = np.uint8(255 * heatmap)
    if cv2 is not None:
        color_map = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    try:
        from matplotlib import colormaps

        return np.uint8(255 * colormaps["jet"](heatmap)[..., :3])
    except Exception:
        # Lightweight jet-like fallback for inference environments without cv2/matplotlib.
        x = heatmap
        red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        return np.uint8(255 * np.stack([red, green, blue], axis=-1))


def save_gradcam_artifacts(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    out_prefix: Path,
    threshold: float = 0.55,
) -> Dict[str, str]:
    image_rgb = unnormalize_image(image_tensor)
    color_map = heatmap_to_rgb(heatmap)
    overlay = np.clip(0.55 * image_rgb + 0.45 * color_map, 0, 255).astype(np.uint8)

    mask = (heatmap >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        mask = (heatmap >= float(np.quantile(heatmap, 0.85))).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        crop = image_rgb
        lesion = image_rgb
    else:
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        crop = image_rgb[y1 : y2 + 1, x1 : x2 + 1]
        lesion = image_rgb.copy()
        lesion[mask == 0] = (0.25 * lesion[mask == 0]).astype(np.uint8)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(out_prefix.with_name(out_prefix.name + "_input.png"))
    Image.fromarray(overlay).save(out_prefix.with_name(out_prefix.name + "_overlay.png"))
    Image.fromarray(lesion).save(out_prefix.with_name(out_prefix.name + "_lesion.png"))
    Image.fromarray(crop).save(out_prefix.with_name(out_prefix.name + "_crop.png"))
    return {
        "input": str(out_prefix.with_name(out_prefix.name + "_input.png")),
        "overlay": str(out_prefix.with_name(out_prefix.name + "_overlay.png")),
        "lesion": str(out_prefix.with_name(out_prefix.name + "_lesion.png")),
        "crop": str(out_prefix.with_name(out_prefix.name + "_crop.png")),
    }


def make_eval_transform(img_size: int):
    return build_eval_transform(img_size)


def load_checkpoint(checkpoint_path: Path, pretrained: bool, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_key = ckpt["model_key"]
    model = MODEL_REGISTRY[model_key](len(ckpt["species_names"]), len(ckpt["disease_names"]), pretrained=pretrained).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def run_single_inference(
    checkpoint_path: Path,
    image_path: Path,
    out_dir: Path,
    mc_passes: int,
    pretrained: bool,
    device: torch.device,
):
    model, ckpt = load_checkpoint(checkpoint_path, pretrained=pretrained, device=device)
    img_size = int(ckpt["img_size"])
    transform = make_eval_transform(img_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        species_logits, disease_logits = model(tensor)
        species_probs = F.softmax(species_logits, dim=1).cpu().numpy()[0]
        disease_probs = F.softmax(disease_logits, dim=1).cpu().numpy()[0]

    result = {
        "species_prediction": ckpt["species_names"][int(species_probs.argmax())],
        "species_confidence": float(species_probs.max()),
        "disease_prediction": ckpt["disease_names"][int(disease_probs.argmax())],
        "disease_confidence": float(disease_probs.max()),
    }

    if isinstance(model, (UnifiedSGCAUncertaintyModel, SGCACrossAttentionUncertaintyGradCAMModel)):
        temp_ds = [(tensor.squeeze(0).cpu(), 0, int(disease_probs.argmax()), str(image_path))]
        loader = DataLoader(temp_ds, batch_size=1, shuffle=False)
        mc = mc_dropout_predict(model, loader, device=device, passes=mc_passes)
        result["predictive_entropy"] = float(mc["pred_entropy"][0])
        result["mutual_information"] = float(mc["mutual_info"][0])
        gradcam = GradCAMExporter(model)
        heatmap = gradcam.generate(tensor, class_idx=int(disease_probs.argmax()))
        artifacts = save_gradcam_artifacts(tensor.squeeze(0), heatmap, out_dir / image_path.stem)
        result["gradcam_artifacts"] = artifacts

    with open(out_dir / f"{image_path.stem}_prediction.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def save_reports(metrics: Dict[str, np.ndarray], species_names: List[str], disease_names: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "species_report.txt", "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                metrics["y_sp"],
                metrics["p_sp"],
                labels=list(range(len(species_names))),
                target_names=species_names,
                zero_division=0,
            )
        )
    with open(out_dir / "disease_report.txt", "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                metrics["y_dis"],
                metrics["p_dis"],
                labels=list(range(len(disease_names))),
                target_names=disease_names,
                zero_division=0,
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Unified SGCA training and inference pipeline.")
    parser.add_argument("--mode", choices=["train", "compare", "infer"], default="train")
    parser.add_argument("--model-key", default="unified_sgca_uncertainty", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data-dir", default="balanced_data")
    parser.add_argument("--split-root", default="", help="Optional root containing train/val/test folders.")
    parser.add_argument("--results-dir", default="sgca_unified_results")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=299)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--augmentation-policy",
        choices=["standard", "robust", "internet_robust", "augmix"],
        default="standard",
    )
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--class-balanced-loss", action="store_true")
    parser.add_argument("--cb-beta", type=float, default=0.9999)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixstyle-p", type=float, default=0.0)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.3)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    if args.mode == "infer":
        if not args.checkpoint or not args.image:
            raise ValueError("--checkpoint and --image are required for infer mode.")
        run_single_inference(
            checkpoint_path=Path(args.checkpoint),
            image_path=Path(args.image),
            out_dir=results_dir,
            mc_passes=args.mc_passes,
            pretrained=args.pretrained,
            device=device,
        )
        return

    if args.split_root:
        split_root = Path(args.split_root)
        data = build_dataloaders_from_split_dirs(
            train_dir=split_root / "train",
            val_dir=split_root / "val",
            test_dir=split_root / "test",
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augmentation_policy=args.augmentation_policy,
            use_weighted_sampler=args.use_weighted_sampler,
        )
    else:
        data = build_dataloaders(
            data_dir=Path(args.data_dir),
            img_size=args.img_size,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )

    with open(results_dir / "dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "split_root": args.split_root,
                "augmentation_policy": args.augmentation_policy,
                "use_weighted_sampler": args.use_weighted_sampler,
                "class_balanced_loss": args.class_balanced_loss,
                "cb_beta": args.cb_beta,
                "label_smoothing": args.label_smoothing,
                "mixstyle_p": args.mixstyle_p,
                "mixstyle_alpha": args.mixstyle_alpha,
                "species_names": data.species_names,
                "disease_names": data.disease_names,
                "train_size": len(data.train_df),
                "val_size": len(data.val_df),
                "test_size": len(data.test_df),
            },
            f,
            indent=2,
        )

    if args.mode == "compare":
        rows = []
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = [args.model_key]
        rows = None

    for model_key in model_keys:
        print("\n" + "=" * 80)
        print("Training", MODEL_DISPLAY_NAMES[model_key])
        print("=" * 80)
        model, history_df, best_path = train_model(
            model_key=model_key,
            data=data,
            device=device,
            results_dir=results_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pretrained=args.pretrained,
            use_amp=use_amp,
            class_balanced_loss=args.class_balanced_loss,
            cb_beta=args.cb_beta,
            label_smoothing=args.label_smoothing,
            mixstyle_p=args.mixstyle_p,
            mixstyle_alpha=args.mixstyle_alpha,
            early_stopping_patience=args.early_stopping_patience,
        )
        print("Best checkpoint:", best_path)
        metrics = evaluate(model, data.test_dl, device)
        save_reports(metrics, data.species_names, data.disease_names, results_dir / MODEL_DISPLAY_NAMES[model_key])
        pd.DataFrame(history_df).to_csv(results_dir / f"{MODEL_DISPLAY_NAMES[model_key]}_history.csv", index=False)

        if rows is not None:
            row = {
                "model_key": model_key,
                "model_name": MODEL_DISPLAY_NAMES[model_key],
                "species_acc": round(metrics["species_acc"], 4),
                "species_f1": round(metrics["species_f1"], 4),
                "disease_acc": round(metrics["disease_acc"], 4),
                "disease_f1": round(metrics["disease_f1"], 4),
            }
            if model_key == "unified_sgca_uncertainty":
                mc = mc_dropout_predict(model, data.test_dl, device=device, passes=args.mc_passes)
                row["mc_acc"] = round(float(accuracy_score(mc["labels"], mc["preds"])), 4)
                row["mc_macro_f1"] = round(float(f1_score(mc["labels"], mc["preds"], average="macro", zero_division=0)), 4)
                row["mean_entropy"] = round(float(mc["pred_entropy"].mean()), 4)
                row["mean_mutual_info"] = round(float(mc["mutual_info"].mean()), 4)
            rows.append(row)

    if rows is not None:
        comparison_df = pd.DataFrame(rows)
        comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)
        print("\nComparison summary")
        print(comparison_df)


if __name__ == "__main__":
    main()
