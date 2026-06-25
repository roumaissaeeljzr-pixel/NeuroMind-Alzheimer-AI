from pathlib import Path
import importlib
import os
import re
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage as ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from cnn_model import (
    InflatedResNet3D,
    FocalLoss,
    crop_to_brain,
    resize_volume,
    normalize_volume,
    TARGET_SHAPE
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
SKULL_STRIPPED_DIR = DATA_DIR / "ADNI_SKULL_STRIPPED"
CSV_PATH = DATA_DIR / "dataset_preprocessed.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_resnet3d_model.pth"

OUTPUT_DIR = BASE_DIR / "outputs"

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


def require_nibabel():
    try:
        return importlib.import_module("nibabel")
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for loading .nii/.nii.gz MRI files. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc


def require_pyplot():
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for saving confusion matrix figures. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc


# ============================================================
# Dataset
# ============================================================

class ADNI3DDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            nib = require_nibabel()
            img = nib.load(str(self.paths[idx]))
            img = nib.as_closest_canonical(img)
            data = img.get_fdata().astype(np.float32)
        except Exception:
            data = np.zeros(TARGET_SHAPE, dtype=np.float32)

        data = crop_to_brain(data)
        data = resize_volume(data, target_shape=TARGET_SHAPE)

        if self.augment:
            if random.random() > 0.5:
                data = np.flip(data, axis=0).copy()

            if random.random() > 0.3:
                data = ndimage.rotate(
                    data,
                    random.uniform(-5, 5),
                    axes=(0, 1),
                    reshape=False,
                    order=1
                )

            if random.random() > 0.5:
                data = data * random.uniform(0.8, 1.2)

            if random.random() > 0.5:
                data = data + np.random.normal(
                    0,
                    0.05,
                    data.shape
                ).astype(np.float32)

        data = normalize_volume(data)

        tensor = torch.from_numpy(data).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return tensor, label


# ============================================================
# Build dataset
# ============================================================

def build_dataset():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not SKULL_STRIPPED_DIR.exists():
        raise FileNotFoundError(f"MRI folder not found: {SKULL_STRIPPED_DIR}")

    df = pd.read_csv(CSV_PATH)
    df = df[df["original_group"].isin(["AD", "CN"])].copy()

    id_to_class = {}
    id_to_subject = {}

    for _, row in df.iterrows():
        match = re.search(r"I(\d+)", str(row["file_name"]))

        if match:
            img_id = "I" + match.group(1)
            id_to_class[img_id] = row["original_group"]
            id_to_subject[img_id] = row["subject"]

    paths = []
    labels = []
    subjects = []

    class_map = {
        "CN": 0,
        "AD": 1
    }

    nii_files = list(SKULL_STRIPPED_DIR.rglob("*.nii")) + list(SKULL_STRIPPED_DIR.rglob("*.nii.gz"))

    for file_path in nii_files:
        match = re.search(r"I(\d+)", file_path.name)

        if match:
            img_id = "I" + match.group(1)

            if img_id in id_to_class:
                paths.append(file_path)
                labels.append(class_map[id_to_class[img_id]])
                subjects.append(id_to_subject[img_id])

    if len(paths) == 0:
        raise RuntimeError(
            "No MRI files matched the CSV. Check filenames, CSV, and ADNI_SKULL_STRIPPED folder."
        )

    print(f"Total matched MRI volumes: {len(paths)}")
    print(f"Unique subjects: {len(set(subjects))}")

    return np.array(paths), np.array(labels), np.array(subjects)


# ============================================================
# Evaluation
# ============================================================

def evaluate_with_tta(model, loader, criterion, split_name="Validation"):
    model.eval()

    all_preds = []
    all_trues = []
    all_probs = []

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            X_flipped = torch.flip(X, dims=[2])

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(DEVICE.type == "cuda")
            ):
                logits_normal = model(X)
                probs_normal = torch.softmax(logits_normal.float(), dim=1)

                logits_flipped = model(X_flipped)
                probs_flipped = torch.softmax(logits_flipped.float(), dim=1)

                probs_final = (probs_normal + probs_flipped) / 2.0

                loss = criterion(logits_normal, y)

            total_loss += loss.item() * X.size(0)
            total_correct += (probs_final.argmax(1) == y).sum().item()
            total_count += y.size(0)

            all_preds.extend(probs_final.argmax(1).cpu().numpy())
            all_trues.extend(y.cpu().numpy())
            all_probs.extend(probs_final[:, 1].cpu().numpy())

    acc = accuracy_score(all_trues, all_preds) * 100
    f1 = f1_score(all_trues, all_preds, average="weighted") * 100
    auc = roc_auc_score(all_trues, all_probs) * 100
    cm = confusion_matrix(all_trues, all_preds)

    print(f"\nResults on {split_name}:")
    print(f"Accuracy : {acc:.2f}%")
    print(f"F1-score : {f1:.2f}%")
    print(f"AUC      : {auc:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            all_trues,
            all_preds,
            target_names=["CN", "AD"]
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt = require_pyplot()

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["CN", "AD"]
    )
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")

    plt.title(
        f"Confusion Matrix - {split_name}\n"
        f"Acc: {acc:.1f}% | F1: {f1:.1f}% | AUC: {auc:.1f}%"
    )
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"confusion_matrix_{split_name.lower()}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved: {output_path}")

    return acc, f1, auc


# ============================================================
# Train
# ============================================================

def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CNN MODULE - INFLATED RESNET3D-18")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"CSV: {CSV_PATH}")
    print(f"MRI folder: {SKULL_STRIPPED_DIR}")
    print(f"Model output: {MODEL_PATH}")

    all_paths, all_labels, all_subjects = build_dataset()

    subject_to_label = {
        s: l for s, l in zip(all_subjects, all_labels)
    }

    unique_subjects = np.array(list(subject_to_label.keys()))
    unique_labels = np.array(
        [subject_to_label[s] for s in unique_subjects]
    )

    trainval_subj, test_subj = train_test_split(
        unique_subjects,
        test_size=0.10,
        random_state=42,
        stratify=unique_labels
    )

    trainval_labels = np.array(
        [subject_to_label[s] for s in trainval_subj]
    )

    train_subj, val_subj = train_test_split(
        trainval_subj,
        test_size=0.111,
        random_state=42,
        stratify=trainval_labels
    )

    assert len(set(train_subj) & set(val_subj)) == 0
    assert len(set(train_subj) & set(test_subj)) == 0
    assert len(set(val_subj) & set(test_subj)) == 0

    train_mask = np.isin(all_subjects, train_subj)
    val_mask = np.isin(all_subjects, val_subj)
    test_mask = np.isin(all_subjects, test_subj)

    print(
        f"Split volumes - Train: {train_mask.sum()} | "
        f"Val: {val_mask.sum()} | Test: {test_mask.sum()}"
    )

    train_ds = ADNI3DDataset(
        all_paths[train_mask],
        all_labels[train_mask],
        augment=True
    )

    val_ds = ADNI3DDataset(
        all_paths[val_mask],
        all_labels[val_mask],
        augment=False
    )

    test_ds = ADNI3DDataset(
        all_paths[test_mask],
        all_labels[test_mask],
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=NUM_WORKERS
    )

    model = InflatedResNet3D(
        num_classes=2,
        pretrained=True
    ).to(DEVICE)

    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(DEVICE.type == "cuda")
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_acc = 0.0

    print("\nTraining started...")
    print(
        f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<10} | "
        f"{'Val Loss':<10} | {'Val Acc':<8}"
    )
    print("-" * 70)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(DEVICE.type == "cuda")
            ):
                logits = model(X)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * X.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += X.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total * 100

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(DEVICE.type == "cuda")
                ):
                    logits = model(X)
                    loss = criterion(logits, y)

                val_loss += loss.item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total * 100

        scheduler.step(epoch_val_acc)

        marker = ""

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " * best saved"

        print(
            f"{epoch:02d}/{EPOCHS:<5} | "
            f"{epoch_train_loss:<12.4f} | "
            f"{epoch_train_acc:<10.2f} | "
            f"{epoch_val_loss:<10.4f} | "
            f"{epoch_val_acc:<8.2f}{marker}"
        )

    total_time = (time.time() - start_time) / 60

    print("\nTraining finished.")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Best model saved at: {MODEL_PATH}")

    print("\nLoading best model for final evaluation...")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    evaluate_with_tta(
        model,
        val_loader,
        criterion,
        split_name="Validation"
    )

    print("\nFinal evaluation on test set...")
    evaluate_with_tta(
        model,
        test_loader,
        criterion,
        split_name="Test"
    )


if __name__ == "__main__":
    train()
