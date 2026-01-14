import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score


# ========================
# Dataset classes
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
        img_name = row.iloc[0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            image = self.transform(image)
        return image, labels


class RetinaTestDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row.iloc[0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


# ========================
# Model builder
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
# Task 2.2: Class-Balanced BCEWithLogits (multi-label)
# ========================
class ClassBalancedBCEWithLogitsLoss(nn.Module):
    """
    Class-Balanced Loss (Cui et al. style effective number) applied to multi-label BCE.
    - Computes per-class weights from positive counts: w_c = (1-beta)/(1-beta^n_c)
    - Normalizes weights to have mean 1 (stability)
    - Applies weights to BCE per class: loss = mean( w_c * BCE(logits, targets) )
    """
    def __init__(self, pos_counts, beta=0.99, reduction="mean", eps=1e-8):
        super().__init__()
        self.beta = float(beta)
        self.reduction = reduction
        self.eps = float(eps)

        pos_counts = np.asarray(pos_counts, dtype=np.float32)
        pos_counts = np.maximum(pos_counts, 1.0)  # avoid div-by-zero

        effective_num = 1.0 - np.power(self.beta, pos_counts)
        weights = (1.0 - self.beta) / (effective_num + self.eps)

        # Normalize to mean=1 so loss scale stays similar to BCE
        weights = weights / (weights.mean() + self.eps)

        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, targets):
        targets = targets.type_as(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B,C]
        w = self.class_weights.view(1, -1).to(logits.device)                                     # [1,C]
        loss = bce * w

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ========================
# Threshold tuning on VAL (per class)
# ========================
def tune_thresholds_from_val(model, val_loader, device, backbone):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    thresholds = []
    for c in range(y_true.shape[1]):
        best_t = 0.5
        best_f1 = -1.0

        if backbone == "efficientnet":
            t_start, t_end = 0.25, 0.86
        else:
            t_start, t_end = 0.20, 0.81

        for t in np.arange(t_start, t_end, 0.01):
            y_pred_c = (probs[:, c] > t).astype(int)
            f1c = f1_score(y_true[:, c], y_pred_c, zero_division=0)
            if f1c > best_f1:
                best_f1 = f1c
                best_t = float(t)

        thresholds.append(best_t)

    return np.array(thresholds, dtype=np.float32)


# ========================
# Test-time augmentation (TTA): original + horizontal flip
# ========================
def predict_probs_tta(model, imgs):
    logits1 = model(imgs)
    p1 = torch.sigmoid(logits1)

    imgs_flip = torch.flip(imgs, dims=[3])
    logits2 = model(imgs_flip)
    p2 = torch.sigmoid(logits2)

    return (p1 + p2) / 2.0


# ========================
# Helper: compute positive class counts from train CSV
# ========================
def compute_pos_counts(train_csv):
    df = pd.read_csv(train_csv)
    # assumes columns: id, D, G, A in that order (matches your pipeline)
    y = df.iloc[:, 1:].values.astype(np.float32)
    pos_counts = y.sum(axis=0)  # [C]
    return pos_counts


# ========================
# Task 2.2 runner (Class-Balanced Loss)
# ========================
def task2_2_class_balanced(backbone, pretrained_ckpt,
                           train_csv, val_csv, offsite_csv, onsite_csv,
                           train_dir, val_dir, offsite_dir, onsite_dir,
                           epochs=40, batch_size=32, img_size=256,
                           beta=0.99):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Task 2.2: Class-Balanced Loss (all-layer fine-tune) ===")
    print(f"Backbone: {backbone} | Device: {device}")

    # Transforms (keep your proven setup)
    if backbone == "efficientnet":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = RetinaMultiLabelDataset(train_csv, train_dir, train_transform)
    val_ds   = RetinaMultiLabelDataset(val_csv,   val_dir,   eval_transform)
    offsite_ds = RetinaMultiLabelDataset(offsite_csv, offsite_dir, eval_transform)
    onsite_ds = RetinaTestDataset(onsite_csv, onsite_dir, eval_transform)

    num_workers = 0
    loader_args = {"batch_size": batch_size, "num_workers": num_workers}

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_args)
    offsite_loader = DataLoader(offsite_ds, shuffle=False, **loader_args)
    onsite_loader  = DataLoader(onsite_ds,  shuffle=False, **loader_args)

    model = build_model(backbone, num_classes=3).to(device)
    state_dict = torch.load(pretrained_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained state_dict: {pretrained_ckpt}")

    for p in model.parameters():
        p.requires_grad = True
    print("All layers unfrozen — full fine-tuning")

    # LR (keep your known-good values)
    if backbone == "resnet18":
        lr = 1e-5
        print(f"Using LR = {lr} for ResNet18")
    else:
        lr = 1e-5
        print(f"Using LR = {lr} for EfficientNet")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_counts = compute_pos_counts(train_csv)
    print(f"Positive counts (D,G,A): {pos_counts} | beta={beta}")
    criterion = ClassBalancedBCEWithLogitsLoss(pos_counts=pos_counts, beta=beta, reduction="mean")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True, min_lr=1e-7
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = f"checkpoints/task2_2_best_{backbone}.pt"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print("   → New best model saved!")

    # Load best for evaluation + submission
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    tuned_thresholds = tune_thresholds_from_val(model, val_loader, device, backbone)
    print(f"\nTuned thresholds (DR, Glaucoma, AMD): {tuned_thresholds}")

    # Offsite evaluation (TTA + tuned thresholds)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in offsite_loader:
            imgs = imgs.to(device)
            probs = predict_probs_tta(model, imgs).cpu().numpy()
            preds = (probs > tuned_thresholds).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    disease_names = ["DR", "Glaucoma", "AMD"]
    print("\n=== Offsite Test Results (Task 2.2 - Class-Balanced) ===")
    per_disease = {}
    for i, name in enumerate(disease_names):
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec  = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1v  = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        acc  = accuracy_score(y_true[:, i], y_pred[:, i])
        kappa = cohen_kappa_score(y_true[:, i], y_pred[:, i])
        per_disease[name] = {"precision": prec, "recall": rec, "f1": f1v}
        print(f"{name}:")
        print(f"   Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1v:.4f} | Acc: {acc:.4f} | Kappa: {kappa:.4f}")

    avg_f1 = np.mean([per_disease[d]["f1"] for d in disease_names])
    print(f"\nAverage F-score (offsite): {avg_f1*100:.1f}%")

    # Onsite submission (TTA + tuned thresholds)
    submission_rows = []
    with torch.no_grad():
        for imgs, img_ids in onsite_loader:
            imgs = imgs.to(device)
            probs = predict_probs_tta(model, imgs).cpu().numpy()
            preds = (probs > tuned_thresholds).astype(int)
            for img_id, pred in zip(img_ids, preds):
                submission_rows.append([img_id, int(pred[0]), int(pred[1]), int(pred[2])])

    sub_df = pd.DataFrame(submission_rows, columns=["id", "D", "G", "A"])
    sub_df = sub_df.sort_values("id").reset_index(drop=True)

    submission_path = f"task2_2_{backbone}_onsite_submission.csv"
    sub_df.to_csv(submission_path, index=False)
    print(f"\nOnsite submission saved: {submission_path}")

    # Save required checkpoint(s):
    
    model_out = f"ati_task2-2_{backbone}.pt"
    torch.save(model.state_dict(), model_out)
    print(f"Saved model checkpoint: {model_out}")

    return avg_f1, per_disease, best_val_loss, model_out


# ========================
# Main — run BOTH backbones for Task 2.2
# ========================
if __name__ == "__main__":
    paths = {
        "train_csv": "train.csv",
        "val_csv":   "val.csv",
        "offsite_csv": "offsite_test.csv",
        "onsite_csv": "onsite_test_submission.csv",
        "train_dir": "./images/train",
        "val_dir":   "./images/val",
        "offsite_dir": "./images/offsite_test",
        "onsite_dir": "./images/onsite_test",
    }

    # CB-loss hyperparam
    beta = 0.999

    configs = [
        ("resnet18",     "./pretrained_backbone/ckpt_resnet18_ep50.pt", 40),
        #("efficientnet", "./pretrained_backbone/ckpt_efficientnet_ep50.pt", 60),
    ]

    global_best_val = float("inf")
    global_best_path = None

    for backbone, ckpt, ep in configs:
        print("\n" + "=" * 60)
        _, _, best_val_loss, model_path = task2_2_class_balanced(
            backbone=backbone,
            pretrained_ckpt=ckpt,
            epochs=ep,
            batch_size=32,
            img_size=256,
            beta=beta,
            **paths
        )

        # Keep a single required filename pointing to the best (by val loss) across both models
        if best_val_loss < global_best_val:
            global_best_val = best_val_loss
            global_best_path = model_path

            state = torch.load(global_best_path, map_location="cpu")
            torch.save(state, "ati_task2-2.pt")
            print(f"Updated global best checkpoint: ati_task2-2.pt (from {global_best_path})")