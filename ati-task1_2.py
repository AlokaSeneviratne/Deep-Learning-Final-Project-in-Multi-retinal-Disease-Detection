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
# Training function for Task 1.2 (frozen backbone)
# ========================
def task1_2_frozen_classifier(backbone, pretrained_ckpt,
                              train_csv, val_csv, offsite_csv, onsite_csv,
                              train_dir, val_dir, offsite_dir, onsite_dir,
                              epochs=15, batch_size=32, img_size=256):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Task 1.2: Frozen backbone, fine-tune classifier only ===")
    print(f"Backbone: {backbone} | Device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_ds = RetinaMultiLabelDataset(train_csv, train_dir, transform)
    val_ds   = RetinaMultiLabelDataset(val_csv,   val_dir,   transform)
    offsite_ds = RetinaMultiLabelDataset(offsite_csv, offsite_dir, transform)
    onsite_ds = RetinaTestDataset(onsite_csv, onsite_dir, transform)

    # Windows-safe DataLoader
    num_workers = 0
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
       # "pin_memory": True,
        #"prefetch_factor": 2,
    }

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_args)
    offsite_loader = DataLoader(offsite_ds, shuffle=False, **loader_args)
    onsite_loader  = DataLoader(onsite_ds,  shuffle=False, **loader_args)

    print(f"Using {num_workers} DataLoader workers (Windows-safe)")

    # Load pretrained model
    model = build_model(backbone, num_classes=3).to(device)
    state_dict = torch.load(pretrained_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained state_dict: {pretrained_ckpt}")

    # Freeze backbone
    for name, param in model.named_parameters():
        if "fc" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (classifier only): {trainable_params}")
    print("Backbone frozen — only classifier head is trainable")

    # === DIFFERENT LEARNING RATES FOR EACH MODEL ===
    if backbone == "resnet18":
        lr = 6.5e-4          # Higher LR for ResNet18 to reach ~61.4%
        print(f"Using LR = {lr} for ResNet18")
    else:  # efficientnet
        lr = 3.3e-4        # Balanced for EfficientNet to maintain ~76% 3.58e3.6e3.65e
        print(f"Using LR = {lr} for EfficientNet")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_ckpt_path = f"checkpoints/task1_2_best_{backbone}_frozen.pt"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"   → New best model saved!")

    # Load best model and evaluate on offsite test
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in offsite_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    disease_names = ["DR", "Glaucoma", "AMD"]
    print("\n=== Offsite Test Results (Task 1.2) ===")
    per_disease = {}
    for i, name in enumerate(disease_names):
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec  = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1   = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        acc  = accuracy_score(y_true[:, i], y_pred[:, i])
        kappa = cohen_kappa_score(y_true[:, i], y_pred[:, i])

        per_disease[name] = {"precision": prec, "recall": rec, "f1": f1}
        print(f"{name}:")
        print(f"   Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Kappa: {kappa:.4f}")

    avg_f1 = np.mean([per_disease[d]["f1"] for d in disease_names])
    print(f"\nAverage F-score (offsite): {avg_f1*100:.1f}%")

    # Generate onsite submission
    submission_rows = []
    model.eval()
    with torch.no_grad():
        for imgs, img_ids in onsite_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            for img_id, pred in zip(img_ids, preds):
                submission_rows.append([img_id, int(pred[0]), int(pred[1]), int(pred[2])])

    sub_df = pd.DataFrame(submission_rows, columns=["id", "D", "G", "A"])
    sub_df = sub_df.sort_values("id").reset_index(drop=True)
    submission_path = f"task1_2_{backbone}_frozen_onsite_submission.csv"
    sub_df.to_csv(submission_path, index=False)
    print(f"\nOnsite submission saved: {submission_path}")
    print("→ Upload this file to Kaggle to get your onsite average F-score")

    return avg_f1, per_disease


# ========================
# Main — run both backbones with different LRs
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

    configs = [
        #("resnet18",     "./pretrained_backbone/ckpt_resnet18_ep50.pt"),
        ("efficientnet", "./pretrained_backbone/ckpt_efficientnet_ep50.pt"),
    ]

    for backbone, ckpt in configs:
        print("\n" + "="*60)
        task1_2_frozen_classifier(
            backbone=backbone,
            pretrained_ckpt=ckpt,
            epochs=7,
            batch_size=32,
            img_size=256,
            **paths
        )