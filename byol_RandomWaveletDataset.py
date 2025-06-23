import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import copy
import pywt
from PIL import Image

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr = 1e-3
pretrain_epochs = 100
finetune_epochs = 50
label_fraction = 0.1  # 1%, 10%, 50% possible

# --- AUGMENTATIONS ---
byol_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

linear_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- WAVELET UTILS ---
def apply_random_LL(img, wavelet='db4'):
    img = np.array(img.convert('L'))  # grayscale
    level = np.random.choice([1, 2, 3])  # random LL level
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    LL = coeffs[0]
    LL_resized = Image.fromarray(np.uint8(LL)).resize((32, 32))
    return LL_resized.convert('RGB')

# --- DATASET CLASSES ---
class BYOLDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return self.transform(img), self.transform(img)

    def __len__(self):
        return len(self.dataset)

class RandomWaveletDataset(Dataset):
    def __init__(self, base_dataset, wavelet='db4', transform=None):
        self.base_dataset = base_dataset
        self.wavelet = wavelet
        self.transform = transform
        self.targets = base_dataset.targets  # ✅ nécessaire pour get_subset_per_class

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = apply_random_LL(img, self.wavelet)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.base_dataset)

# --- MLP HEAD ---
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

# --- BYOL ARCHITECTURE ---
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

# --- LOAD CIFAR10 BASE DATA ---
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True)
byol_loader = DataLoader(BYOLDataset(cifar_train, byol_transform), batch_size=batch_size, shuffle=True)

# --- BACKBONE ---
backbone = models.resnet18(weights=None)
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
backbone.maxpool = nn.Identity()
backbone.fc = nn.Identity()

# --- BYOL PRETRAIN ---
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

# --- UTILS ---
def get_subset_per_class(dataset, fraction):
    targets = np.array(dataset.targets)
    indices = []
    for class_idx in range(10):
        class_indices = np.where(targets == class_idx)[0]
        n_samples = max(1, int(len(class_indices) * fraction))
        selected = np.random.choice(class_indices, n_samples, replace=False)
        indices.extend(selected)
    np.random.shuffle(indices)
    return Subset(dataset, indices)

# --- FINE-TUNING ---
base_train = datasets.CIFAR10(root='./data', train=True, transform=None)
wavelet_train = RandomWaveletDataset(base_train, wavelet='db4', transform=linear_transform)
subset_train = get_subset_per_class(wavelet_train, label_fraction)
testset = datasets.CIFAR10(root='./data', train=False, transform=linear_transform)

trainloader = DataLoader(subset_train, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size)

class FineTunedClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
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

# --- FINAL TEST ---
finetune_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        pred = finetune_model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"\n🧪 Final Test Accuracy: {correct / total:.4f}")
