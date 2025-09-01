import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import random

# ============ CONFIG ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
learning_rate = 0.001
num_epochs = 100
use_extended_dataset = False  # 🔁 Change ici pour basculer vers le dataset étendu
val_fraction = 0.5
random_seed = 42
val_split_seed = 123

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# ============ TRANSFORMATIONS ============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ============ DATASETS ============
if use_extended_dataset:
    saved = torch.load('./data/cifar10_extended.pt')
    data = saved['data']
    labels = saved['labels']
    train_set = TensorDataset(data, labels)
else:
    raw_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_set = raw_train

# Validation depuis testset
full_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_size = int(len(full_testset) * val_fraction)
test_size = len(full_testset) - val_size
val_set, final_test_set = random_split(full_testset, [val_size, test_size], generator=torch.Generator().manual_seed(val_split_seed))

# ============ MODEL ============
def get_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

# ============ TRAIN & EVAL FUNCTIONS ============
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

# ============ RUN ============
model = get_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False)

print("\n🚀 Entraînement classique ResNet-18 sur CIFAR-10")

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion)
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ============ TEST FINAL ============
test_loss, test_acc = eval_epoch(model, test_loader, criterion)
print(f"\n🎯 Résultats finaux sur le testset : Loss = {test_loss:.4f} | Accuracy = {test_acc:.4f}")
