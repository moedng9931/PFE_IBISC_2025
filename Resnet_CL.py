import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import pywt
import numpy as np
from PIL import Image
import random

# ============ CONFIG ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
learning_rate = 0.001
num_epochs_final = 100
patience = 10
use_extended_dataset = False  # utiliser le dataset étendu ou non
val_fraction = 0.5
random_seed = 42
val_split_seed = 123

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# ============ TRANSFORMATIONS ============
transform_original = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ============ WAVELET TRANSFORM ============

def wavelet_LL1(img_pil, wavelet_name='bior3.3'):
    img = np.array(img_pil).astype(np.float32) / 255.0
    channels_LL1 = []
    for c in range(3):
        cA1, _ = pywt.dwt2(img[:, :, c], wavelet=wavelet_name)
        LL1 = Image.fromarray(np.uint8(cA1 * 255)).resize((32, 32))
        channels_LL1.append(np.array(LL1))
    merged = np.stack(channels_LL1, axis=2)
    img_tensor = torch.from_numpy(merged.transpose((2, 0, 1))).float() / 255.0
    return (img_tensor - 0.5) / 0.5

# ============ DATASETS ============
if use_extended_dataset:
    saved = torch.load('./data/cifar10_extended.pt')
    data = saved['data']
    labels = saved['labels']
    train_set = TensorDataset(data, labels)
else:
    raw_train = datasets.CIFAR10(root='./data', train=True, download=True)
    images = []
    labels = []
    for img, label in raw_train:
        img_tensor = transform_original(img)
        images.append(img_tensor)
        labels.append(label)
    train_set = TensorDataset(torch.stack(images), torch.tensor(labels))

# Validation depuis testset
full_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_original)
val_size = int(len(full_testset) * val_fraction)
test_size = len(full_testset) - val_size
val_set, final_test_set = random_split(full_testset, [val_size, test_size], generator=torch.Generator().manual_seed(val_split_seed))

# ============ MODEL ============
def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

# ============ TRAINING & EVAL FUNCTIONS ============
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# ============ CURRICULUM LEARNING ============

def train_curriculum(train_dataset, val_loader):
    print("\n🚀 Début du Curriculum Learning (LL1 ➜ Original)")
    stages = ["LL1", "Original"]
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for stage in stages:
        print(f"\n=== Étape: {stage} ===")
        if stage == "LL1":
            def LL1_transform(img):
                return wavelet_LL1(transforms.ToPILImage()(img))
            wavelet_data = torch.stack([LL1_transform(img) for img, _ in train_dataset])
            wavelet_labels = torch.tensor([label for _, label in train_dataset])
            train_loader = DataLoader(TensorDataset(wavelet_data, wavelet_labels), batch_size=batch_size, shuffle=True)

            # Early stopping
            best_val_loss = float('inf')
            counter = 0
            for epoch in range(1, 101):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_acc = eval_epoch(model, val_loader, criterion)
                print(f"[{stage}] Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("⏹️ Early stopping déclenché.")
                        break
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(1, num_epochs_final + 1):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_acc = eval_epoch(model, val_loader, criterion)
                print(f"[{stage}] Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    return model

# ============ TEST FINAL ============
def test_model(model):
    print("\n🎯 Évaluation finale sur le jeu de test")
    test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    print(f"📊 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# ============ RUN ============
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
final_model = train_curriculum(train_set, val_loader)
test_model(final_model)
