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
# Task 3.2 Attention block: lightweight spatial MHA (on 7x7 features)
# ========================
class SpatialMHA(nn.Module):
    """
    Apply Multi-Head Attention over spatial tokens.
    Input:  x (B, C, H, W)
    Tokens: (B, HW, C) -> MHA -> (B, HW, C) -> back to (B, C, H, W)

    Notes:
    - We keep it "safe" by initializing proj_out to 0, so it starts near identity.
    - Dropout helps regularize.
    """
    def __init__(self, channels, num_heads=8, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.proj_out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(proj_dropout),
        )

        # Start close to identity (residual only)
        if isinstance(self.proj_out[0], nn.Linear):
            nn.init.zeros_(self.proj_out[0].weight)
            nn.init.zeros_(self.proj_out[0].bias)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # (B, HW, C)
        tokens_n = self.norm(tokens)
        attn_out, _ = self.mha(tokens_n, tokens_n, tokens_n, need_weights=False)  # (B, HW, C)
        attn_out = self.proj_out(attn_out)
        tokens = tokens + attn_out  # residual

        x = tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x


# ========================
# ResNet18 + MHA (Task 3.2)
# ========================
class ResNet18_MHA(nn.Module):
    def __init__(self, num_classes=3, num_heads=8, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # ResNet18 final feature map channels = 512, spatial = ~7x7 for 224/256 inputs
        self.mha = SpatialMHA(
            channels=512,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

    def forward(self, x):
        # manual forward until layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Attention before pooling
        x = self.mha(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x


# ========================
# Task 3.2 runner (ResNet only)
# ========================
def task3_2_mha_resnet18(
    pretrained_ckpt,
    train_csv, val_csv, offsite_csv, onsite_csv,
    train_dir, val_dir, offsite_dir, onsite_dir,
    epochs=40, batch_size=32, img_size=256,
    num_heads=8, attn_dropout=0.1, proj_dropout=0.1,
    warmup_epochs=1,
    lr=1e-5,
    teamname="ati",
    use_fixed_thresholds=True,
    fixed_thresholds=np.array([0.64, 0.23, 0.39], dtype=np.float32),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== Task 3.2: Multi-Head Attention (MHA) on ResNet18 ===")
    print(f"Device: {device}")
    print(f"Baseline ckpt: {pretrained_ckpt}")
    print(f"MHA: heads={num_heads}, attn_dropout={attn_dropout}, proj_dropout={proj_dropout}")
    print(f"Warmup epochs: {warmup_epochs} | LR: {lr}")

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

    model = ResNet18_MHA(
        num_classes=3,
        num_heads=num_heads,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
    ).to(device)

    # ========================
    # CRITICAL FIX:
    # load baseline weights INTO backbone only (keys match)
    # ========================
    state_dict = torch.load(pretrained_ckpt, map_location=device)
    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
    print(f"Loaded baseline checkpoint into backbone: {pretrained_ckpt}")
    print("Backbone load - missing keys:", len(missing), "unexpected keys:", len(unexpected))
    print("Example backbone weight mean:", model.backbone.conv1.weight.mean().item())

    # (optional but safe) don’t zero the classifier bias unless you really want to.
    # Your baseline already learned a good bias. Keep it.

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

    # ========================
    # Train setup (warmup: attention+head only)
    # ========================
    for _, p in model.named_parameters():
        p.requires_grad = True

    if warmup_epochs > 0:
        for name, p in model.named_parameters():
            if name.startswith("backbone."):
                p.requires_grad = False
        print(f"Warmup: training MHA+head only for {warmup_epochs} epochs (backbone frozen)")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True, min_lr=1e-7
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = "checkpoints/task3_2_best_resnet18_mha.pt"

    # ========================
    # Training loop
    # ========================
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

    # Load best model
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    # Thresholds
    tuned_thresholds = tune_thresholds_from_val(model, val_loader, device, backbone="resnet18")
    print(f"\nTuned thresholds (DR, Glaucoma, AMD): {tuned_thresholds}")

    if use_fixed_thresholds:
        tuned_thresholds = np.array(fixed_thresholds, dtype=np.float32)
        print(f"Using FIXED thresholds (DR, Glaucoma, AMD): {tuned_thresholds}")

    # ========================
    # Offsite evaluation (TTA + thresholds)
    # ========================
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
    print("\n=== Offsite Test Results (Task 3.2 - MHA) ===")
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

    # ========================
    # Onsite submission CSV (TTA + thresholds)
    # ========================
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

    submission_path = "task3_2_resnet18_mha_onsite_submission.csv"
    sub_df.to_csv(submission_path, index=False)
    print(f"\nOnsite submission saved: {submission_path}")
    print("→ Upload this file to Kaggle to get your onsite average F-score")

    # Save final model with required naming scheme
    final_model_path = f"{teamname}_task3-2.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final MHA model: {final_model_path}")

    return avg_f1, per_disease


# ========================
# Main
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

    # Use your strongest baseline (you said Task 2.2 is best)
    BASELINE_CKPT = "./checkpoints/task2_2_best_resnet18.pt"

    # Keep the fixed thresholds you used successfully before (change here if you want)
    FIXED_THRESHOLDS = np.array([0.64, 0.23, 0.39], dtype=np.float32)

    task3_2_mha_resnet18(
        pretrained_ckpt=BASELINE_CKPT,
        epochs=40,
        batch_size=32,
        img_size=256,
        num_heads=8,          # try 4 or 8 first
        attn_dropout=0.10,    # try 0.05 if over-regularized
        proj_dropout=0.10,
        warmup_epochs=1,      # 1–3 is usually enough
        lr=1e-5,
        teamname=TEAMNAME,
        use_fixed_thresholds=True,
        fixed_thresholds=FIXED_THRESHOLDS,
        **paths
    )
