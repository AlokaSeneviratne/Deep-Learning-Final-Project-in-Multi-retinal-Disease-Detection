import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# ========================
# Compute dataset mean/std (run once on train set)
# ========================
def compute_dataset_stats(csv_file, image_dir, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ds = RetinaMultiLabelDataset(csv_file, image_dir, transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    mean = 0.
    std = 0.
    n_samples = 0.
    for imgs, _ in loader:
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples
    return mean.tolist(), std.tolist()

# Example usage: Uncomment to compute and print (replace paths)
# train_csv = "train.csv"
# train_image_dir = "./images/train"
# mean, std = compute_dataset_stats(train_csv, train_image_dir)
# print(f"ODIR Train Mean: {mean}, Std: {std}")

# ========================
# Dataset classes (unchanged)
# ========================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.image_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels

class RetinaTestDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.image_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_id

# ========================
# Build model (unchanged)
# ========================
def build_model(backbone="resnet18", num_classes=3):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model

# ========================
# Load checkpoint (unchanged)
# ========================
def load_checkpoint_strict(model, ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Missing keys   : {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    assert len(missing) == 0 and len(unexpected) == 0, (
        "Checkpoint mismatch. Fix model definition."
    )

    model.to(device)
    model.eval()
    return model

# ========================
# Offsite metrics (unchanged)
# ========================
def infer_offsite_and_metrics(model, loader, threshold=0.5):
    y_true_chunks, y_pred_chunks = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(next(model.parameters()).device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)

            y_true_chunks.append(labels.numpy().astype(int))
            y_pred_chunks.append(preds.astype(int))

    Y_true = np.concatenate(y_true_chunks, axis=0)
    Y_pred = np.concatenate(y_pred_chunks, axis=0)

    disease_names = ["DR", "Glaucoma", "AMD"]
    metrics = {}

    for i, disease in enumerate(disease_names):
        y_t = Y_true[:, i]
        y_p = Y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        metrics[disease] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "kappa": kappa
        }

    average_f1 = np.mean([metrics[d]['f1'] for d in disease_names]) * 100.0  # In %
    return metrics, average_f1

# ========================
# Onsite save (unchanged)
# ========================
def infer_onsite_and_save(model, loader, out_csv_path, threshold=0.5):
    rows = []
    with torch.no_grad():
        for imgs, img_ids in loader:
            imgs = imgs.to(next(model.parameters()).device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)

            for img_id, p in zip(img_ids, preds):
                rows.append([img_id, int(p[0]), int(p[1]), int(p[2])])

    sub = pd.DataFrame(rows, columns=["id", "D", "G", "A"])
    sub.sort_values(by="id", inplace=True)  # Ensure sorted by id for consistency
    sub.to_csv(out_csv_path, index=False)
    print(f"Saved Kaggle submission: {out_csv_path}")

# ========================
# Run Task 1.1
# ========================
def run_task_1_1(backbone, ckpt_path, paths, batch_size=32, img_size=256, threshold=0.5):
    # Force CPU for exact reproducibility
    device = torch.device("cpu")
    # Determinism (if using GPU later, uncomment)
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    print(f"\n✅ Using device: {device}")
    print("============================================================")
    print(f"Running Task 1.1 (No fine-tuning) for: {backbone}")
    print("============================================================")

    # Transform: Use computed ODIR stats (replace with your computed values)
    # Example: mean=[0.42, 0.24, 0.12], std=[0.28, 0.17, 0.11] (common for fundus; compute yours)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Start with ImageNet; replace if needed
    ])

    offsite_ds = RetinaMultiLabelDataset(paths["offsite_csv"], paths["offsite_dir"], transform)
    offsite_loader = DataLoader(offsite_ds, batch_size=batch_size, shuffle=False, num_workers=0)  # num_workers=0 for determinism

    onsite_ds = RetinaTestDataset(paths["onsite_csv"], paths["onsite_dir"], transform)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(backbone, num_classes=3)
    model = load_checkpoint_strict(model, ckpt_path, device)

    per_disease_metrics, offsite_avg_f1 = infer_offsite_and_metrics(model, offsite_loader, threshold=threshold)

    out_csv = f"task1_1_{backbone}_onsite_submission.csv"
    infer_onsite_and_save(model, onsite_loader, out_csv, threshold=threshold)

    pretty = "EfficientNet" if backbone == "efficientnet" else "ResNet18"
    print("\nOffsite Test Metrics (Task 1.1 - No fine-tuning)")
    print("------------------------------------------------")
    print(f"Model: {pretty}")
    for disease, m in per_disease_metrics.items():
        print(f"\n{disease}:")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1-score:  {m['f1']:.4f}")
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  Kappa:    {m['kappa']:.4f}")
    print(f"\nAverage F-score: {offsite_avg_f1:.1f}%")

    return pretty, offsite_avg_f1, per_disease_metrics

# ========================
# Main
# ========================
if __name__ == "__main__":
    paths = {
        "offsite_csv": "offsite_test.csv",
        "onsite_csv": "onsite_test_submission.csv",
        "offsite_dir": "./images/offsite_test",
        "onsite_dir": "./images/onsite_test",
    }

    configs = [
        ("efficientnet", "./pretrained_backbone/ckpt_efficientnet_ep50.pt"),
        ("resnet18", "./pretrained_backbone/ckpt_resnet18_ep50.pt"),
    ]

    results = []
    all_metrics = {}
    for backbone, ckpt_path in configs:
        name, avg_f1, per_metrics = run_task_1_1(
            backbone=backbone,
            ckpt_path=ckpt_path,
            paths=paths,
            batch_size=32,
            img_size=256,
            threshold=0.5
        )
        results.append((name, avg_f1))
        all_metrics[name] = per_metrics

    print("\n==============================")
    print("Final Offsite Summary (Task 1.1)")
    print("==============================")
    print(f"{'Model':<15} {'Avg F-score (%)':>15}")
    for name, avg_f1 in results:
        print(f"{name:<15} {avg_f1:>15.1f}")
    print()
