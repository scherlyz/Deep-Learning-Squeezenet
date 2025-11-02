import os, shutil, time, math
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.preprocessing import label_binarize

# === KONFIGURASI ===
DATA_DIR = "all_data"   # folder utama dataset kamu
WORK_DIR = "dataset_split"  # folder hasil split otomatis
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
SEED = 42

# Buat reproducible
torch.manual_seed(SEED)
np.random.seed(SEED)

# === CEK DATASET ===
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Folder {DATA_DIR} tidak ditemukan!")

classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print("Kelas ditemukan:", classes)

# Kumpulkan semua path dan label
all_paths, all_labels = [], []
for c in classes:
    folder = os.path.join(DATA_DIR, c)
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
            all_paths.append(os.path.join(folder, f))
            all_labels.append(c)

# Split dataset
train_idx, test_idx = train_test_split(
    range(len(all_paths)), test_size=TEST_SPLIT, stratify=all_labels, random_state=SEED
)
train_idx_sub, val_idx_sub = train_test_split(
    train_idx, test_size=VAL_SPLIT, stratify=[all_labels[i] for i in train_idx], random_state=SEED
)

# Buat folder split
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
for split in ["train", "val", "test"]:
    for c in classes:
        os.makedirs(os.path.join(WORK_DIR, split, c), exist_ok=True)

def copy_files(indices, split_name):
    for i in indices:
        src = all_paths[i]
        label = all_labels[i]
        dst = os.path.join(WORK_DIR, split_name, label, os.path.basename(src))
        shutil.copy2(src, dst)

print("Membagi dataset...")
copy_files(train_idx_sub, "train")
copy_files(val_idx_sub, "val")
copy_files(test_idx, "test")
print("Dataset berhasil dibagi ke folder:", WORK_DIR)

# === TRANSFORMASI GAMBAR ===
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=20, scale=(0.8, 1.0), shear=20),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === DATASET & DATALOADER ===
train_dataset = datasets.ImageFolder(os.path.join(WORK_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(WORK_DIR, "val"), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(WORK_DIR, "test"), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# === MODEL: SqueezeNet ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Menggunakan device:", device)

num_classes = len(train_dataset.classes)
model = models.squeezenet1_1(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(512, num_classes, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1))
)
model.num_classes = num_classes
model = model.to(device)

# === LOSS & OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === TRAINING ===
print("\nMulai training...\n")
start_train = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss, preds_list, labels_list = 0, [], []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb).view(xb.size(0), -1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds_list += out.argmax(1).cpu().numpy().tolist()
        labels_list += yb.cpu().numpy().tolist()
    train_acc = accuracy_score(labels_list, preds_list)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_dataset):.4f} | Acc: {train_acc:.4f}")

print(f"\nWaktu training: {time.time()-start_train:.2f} detik")

# === TESTING ===
print("\nEvaluasi pada data testing...\n")
start_test = time.time()
model.eval()
y_true, y_pred, y_score = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb).view(xb.size(0), -1)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true += yb.numpy().tolist()
        y_pred += preds.tolist()
        y_score += probs.tolist()
test_time = time.time() - start_test

# === METRIK ===
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
try:
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    roc_auc = roc_auc_score(y_true_bin, np.array(y_score), average="macro", multi_class="ovo")
except Exception:
    roc_auc = float("nan")

print("=== HASIL TESTING ===")
print(f"Akurasi      : {acc:.4f}")
print(f"Presisi (macro): {prec:.4f}")
print(f"Recall (macro) : {rec:.4f}")
print(f"F1-Score (macro): {f1:.4f}")
print(f"ROC/AUC        : {roc_auc:.4f}")
print(f"Waktu Testing  : {test_time:.2f} detik")

# === SIMPAN MODEL ===
torch.save(model.state_dict(), "squeezenet_trained.pth")
print("\nModel tersimpan di: squeezenet_trained.pth")
