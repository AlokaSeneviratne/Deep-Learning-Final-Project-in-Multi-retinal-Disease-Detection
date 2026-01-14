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
# SE block (Task 3.1)
# ========================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation:
      - Squeeze: GlobalAvgPool -> (B,C)
      - Excite:  FC(C->C/r->C) + sigmoid
      - Scale:   channel-wise multiplication
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

        # initialize last layer so SE starts close-ish to identity
        if isinstance(self.fc[-2], nn.Linear):
            nn.init.zeros_(self.fc[-2].weight)
            nn.init.constant_(self.fc[-2].bias, 6.0)  # sigmoid(6)=0.9975 ~ identity

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        g = self.fc(s).view(b, c, 1, 1)
        return x * g


# ========================
# SE-wrapped ResNet18
# ========================
class ResNet18_SE(nn.Module):
    def __init__(self, num_classes=3, se_reduction=8):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.se = SEBlock(channels=512, reduction=se_reduction)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.se(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x


# ========================
# SE-wrapped EfficientNet-B0
# ========================
class EfficientNetB0_SE(nn.Module):
    def __init__(self, num_classes=3, se_reduction=8):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        self.se = SEBlock(channels=1280, reduction=se_reduction)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.se(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


# ========================
# Model builder (Task 3.1 SE)
# ========================
def build_model_se(backbone="resnet18", num_classes=3, se_reduction=8):
    if backbone == "resnet18":
        return ResNet18_SE(num_classes=num_classes, se_reduction=se_reduction)
    elif backbone == "efficientnet":
        return EfficientNetB0_SE(num_classes=num_classes, se_reduction=se_reduction)
    else:
        raise ValueError("Unsupported backbone")


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
        best_t, best_f1 = 0.5, -1.0
        t_start, t_end = (0.25, 0.86) if backbone == "efficientnet" else (0.20, 0.81)
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
# Task 3.1: Train/Eval with SE
# ========================
def task3_1_se(backbone, pretrained_ckpt,
               train_csv, val_csv, offsite_csv, onsite_csv,
               train_dir, val_dir, offsite_dir, onsite_dir,
               epochs=40, batch_size=32, img_size=256,
               se_reduction=8,
               teamname="ati",
               warmup_epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Task 3.1: Squeeze-and-Excitation (SE) ===")
    print(f"Backbone: {backbone} | Device: {device}")

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

    model = build_model_se(backbone, num_classes=3, se_reduction=se_reduction).to(device)

    # load from best strong baseline checkpoint
    loaded_ckpt_path = "./checkpoints/task2_2_best_resnet18.pt"
    state_dict = torch.load(loaded_ckpt_path, map_location=device)
    model.backbone.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {loaded_ckpt_path} (strict=False, SE is new)")
    print(f"SE reduction ratio: {se_reduction}")

    if hasattr(model.backbone, "fc") and model.backbone.fc.bias is not None:
        nn.init.zeros_(model.backbone.fc.bias)

    # ========================
    # SANITY CHECK (VERY IMPORTANT)
    # ========================
    model.eval()
    imgs, labels = next(iter(val_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        logits = model(imgs)
        probs = torch.sigmoid(logits)

    print("SANITY CHECK:")
    print("  probs mean:", probs.mean().item())
    print("  probs min :", probs.min().item())
    print("  probs max :", probs.max().item())
    print("  labels mean:", labels.mean().item())


    # Full fine-tuning (with warmup freeze)
    for _, p in model.named_parameters():
        p.requires_grad = True

    if warmup_epochs > 0:
        for name, p in model.named_parameters():
            if name.startswith("backbone."):
                p.requires_grad = False
        print(f"Warmup: training SE+head only for {warmup_epochs} epochs (backbone frozen)")

    lr = 1e-5
    print(f"Using LR = {lr}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-4)

    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True, min_lr=1e-7
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = f"checkpoints/task3_1_best_{backbone}_se.pt"

    for epoch in range(epochs):
        if epoch == warmup_epochs:
            for _, p in model.named_parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr, weight_decay=1e-4)
            print("Warmup done: backbone unfrozen — full fine-tuning continues")

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

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    tuned_thresholds = tune_thresholds_from_val(model, val_loader, device, backbone)
    print(f"\nTuned thresholds (DR, Glaucoma, AMD): {tuned_thresholds}")

    tuned_thresholds = np.array([0.64, 0.23, 0.39], dtype=np.float32)
    print(f"Using FIXED thresholds (DR, Glaucoma, AMD): {tuned_thresholds}")


    # ========================
    # (CHANGE) LOCK THRESHOLDS TO YOUR BEST TASK 2.2 RESNET VALUES
    # ========================
    if backbone == "resnet18":
        tuned_thresholds = np.array([0.62, 0.22, 0.37], dtype=np.float32)
        print(f"Overridden thresholds (ResNet18 locked from Task 2.2): {tuned_thresholds}")

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
    print("\n=== Offsite Test Results (Task 3.1 - SE) ===")
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

    # Onsite submission CSV
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

    submission_path = f"task3_1_{backbone}_se_onsite_submission.csv"
    sub_df.to_csv(submission_path, index=False)
    print(f"\nOnsite submission saved: {submission_path}")
    print("→ Upload this file to Kaggle to get your onsite average F-score")

    final_model_path = f"{teamname}_task3-1.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final SE model: {final_model_path}")

    return avg_f1, per_disease


# ========================
# Main — run Task 3.1 (ResNet only)
# ========================
if __name__ == "__main__":
    TEAMNAME = "ati"

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
        ("resnet18", "./pretrained_backbone/ckpt_resnet18_ep50.pt", 40),
    ]

    for backbone, ckpt, ep in configs:
        print("\n" + "="*60)
        task3_1_se(
            backbone=backbone,
            pretrained_ckpt=ckpt,
            epochs=ep,
            batch_size=32,
            img_size=256,
            se_reduction=32,
            teamname=TEAMNAME,
            warmup_epochs=1,
            **paths
        )
