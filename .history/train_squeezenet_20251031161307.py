import os, shutil, time, math
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# === KONFIGURASI ===
DATA_DIR = "all_data"
WORK_DIR = "dataset_split"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# === CEK DATASET ===
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print("Kelas ditemukan:", classes)

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.classes)
model = models.squeezenet1_1(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(512, num_classes, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === TRAINING LOOP ===
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\nMulai training...\n")
for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb).view(xb.size(0), -1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    train_acc = correct / total
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(train_acc)

    # === VALIDASI ===
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb).view(xb.size(0), -1)
            loss = criterion(out, yb)
            val_loss += loss.item()
            val_correct += (out.argmax(1) == yb).sum().item()
            val_total += yb.size(0)
    val_acc = val_correct / val_total
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | "
          f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")

# === VISUALISASI ===
epochs = np.arange(1, EPOCHS + 1)
plt.figure(figsize=(10, 4))

# Grafik Akurasi
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accs, label="Train Accuracy", marker='o')
plt.plot(epochs, val_accs, label="Validation Accuracy", marker='s')
plt.title("Perbandingan Akurasi")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Grafik Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, val_losses, label="Validation Loss", marker='s')
plt.title("Perbandingan Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

print("\n✅ Grafik training tersimpan di: training_history.png")
torch.save(model.state_dict(), "squeezenet_trained.pth")
