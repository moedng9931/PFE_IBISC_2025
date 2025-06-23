import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import copy

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr = 1e-3
pretrain_epochs = 100
finetune_epochs = 50
label_fraction = 0.1  # 🧪 change ici : 0.01 (1%), 0.1 (10%), 0.5 (50%)

# --- BYOL AUGMENTATION ---
byol_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Dataset Wrapper for BYOL ---
class BYOLDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return self.transform(img), self.transform(img)

    def __len__(self):
        return len(self.dataset)

# --- MLP Head ---
class MLPHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- BYOL Architecture ---
class BYOL(nn.Module):
    def __init__(self, backbone, hidden_dim=1024, out_dim=256, m=0.99):
        super().__init__()
        self.online_encoder = nn.Sequential(backbone, MLPHead(512, hidden_dim, out_dim))
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = MLPHead(out_dim, hidden_dim, out_dim)
        self.m = m

    def update_moving_average(self):
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = self.m * target.data + (1 - self.m) * online.data

    def forward(self, x1, x2):
        z1 = self.predictor(self.online_encoder(x1))
        with torch.no_grad():
            z2 = self.target_encoder(x2)
        z1 = nn.functional.normalize(z1, dim=-1)
        z2 = nn.functional.normalize(z2, dim=-1)
        return 2 - 2 * (z1 * z2).sum(dim=-1).mean()

# --- LOAD CIFAR10 (no labels used for SSL) ---
dataset = datasets.CIFAR10(root='./data', train=True, download=True)
byol_loader = DataLoader(BYOLDataset(dataset, byol_transform), batch_size=batch_size, shuffle=True)

# --- BACKBONE ---
backbone = models.resnet18(weights=None)
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
backbone.maxpool = nn.Identity()
backbone.fc = nn.Identity()

# --- BYOL TRAINING ---
byol_model = BYOL(backbone).to(device)
optimizer = optim.Adam(byol_model.parameters(), lr=lr)

for epoch in range(pretrain_epochs):
    byol_model.train()
    total_loss = 0
    for x1, x2 in byol_loader:
        x1, x2 = x1.to(device), x2.to(device)
        loss = byol_model(x1, x2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        byol_model.update_moving_average()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] BYOL Loss: {total_loss/len(byol_loader):.4f}")

# --- LINEAR TRANSFORM (for probing/fine-tuning) ---
linear_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 canaux pour CIFAR-10
])

# --- DATASET REDUCTION FOR SUPERVISED STAGE ---
def get_subset_per_class(dataset, fraction):
    targets = np.array(dataset.targets)
    indices = []
    for class_idx in range(10):  # 10 classes in CIFAR-10
        class_indices = np.where(targets == class_idx)[0]
        n_samples = max(1, int(len(class_indices) * fraction))
        selected = np.random.choice(class_indices, n_samples, replace=False)
        indices.extend(selected)
    np.random.shuffle(indices)
    return Subset(dataset, indices)

# --- PREPARE DATALOADERS ---
trainset = datasets.CIFAR10(root='./data', train=True, transform=linear_transform)
subset_trainset = get_subset_per_class(trainset, label_fraction)
testset = datasets.CIFAR10(root='./data', train=False, transform=linear_transform)

trainloader = DataLoader(subset_trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size)

# --- FINE-TUNING CLASSIFIER (FULL MODEL) ---
class FineTunedClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # 🔓 tout est entraîné
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc(self.encoder(x))

finetune_model = FineTunedClassifier(byol_model.online_encoder[0]).to(device)
optimizer = optim.Adam(finetune_model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(finetune_epochs):
    finetune_model.train()
    correct, total, total_loss = 0, 0, 0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        out = finetune_model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    print(f"[FineTune Epoch {epoch+1}] Loss: {total_loss:.4f}, Accuracy: {correct/total:.4f}")

# --- TEST FINAL ---
finetune_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        pred = finetune_model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"\n🧪 Final Test Accuracy: {correct / total:.4f}")
