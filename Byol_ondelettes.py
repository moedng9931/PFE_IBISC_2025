import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pywt, copy

# === PARAMÈTRES ===
wavelet_type = "coif1"  # 🔁 Change "haar", "db4", "db8", etc.
batch_size, lr = 128, 1e-3
pretrain_epochs, finetune_epochs = 100, 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === ONDELETTE ===
def apply_wavelet_transform(img, wavelet=wavelet_type, level=1):
    arr = np.array(img)
    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level, mode='periodization')
    arr, _ = pywt.coeffs_to_array(coeffs)
    arr = np.uint8(255 * (arr - arr.min()) / (arr.max() - arr.min()))
    return Image.fromarray(arr).convert("RGB")

class WaveletTransform:
    def __init__(self, base_transform, wavelet=wavelet_type):
        self.base_transform = base_transform
        self.wavelet = wavelet

    def __call__(self, img):
        img = apply_wavelet_transform(img, self.wavelet)
        return self.base_transform(img)

# === AUGMENTATION BYOL ===
base_aug = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
    transforms.RandomGrayscale(0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
wavelet_aug = WaveletTransform(base_aug)

# === DATASET DOUBLE VUE ===
class BYOLDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return self.transform(img), self.transform(img)

    def __len__(self):
        return len(self.dataset)

# === MLP PROJECTOR ===
class MLPHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

# === BYOL MODEL ===
class BYOL(nn.Module):
    def __init__(self, backbone, hidden_dim=1024, out_dim=256, m=0.99):
        super().__init__()
        self.online_encoder = nn.Sequential(backbone, MLPHead(512, hidden_dim, out_dim))
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters(): param.requires_grad = False
        self.predictor = MLPHead(out_dim, hidden_dim, out_dim)
        self.m = m

    def update_moving_average(self):
        for o, t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            t.data = self.m * t.data + (1 - self.m) * o.data

    def forward(self, x1, x2):
        z1 = nn.functional.normalize(self.predictor(self.online_encoder(x1)), dim=-1)
        with torch.no_grad():
            z2 = nn.functional.normalize(self.target_encoder(x2), dim=-1)
        return 2 - 2 * (z1 * z2).sum(dim=-1).mean()

# === CHARGEMENT DES DONNÉES CIFAR-10 ===
dataset = datasets.CIFAR10(root="./data", train=True, download=True)
loader = DataLoader(BYOLDataset(dataset, wavelet_aug), batch_size=batch_size, shuffle=True)

backbone = models.resnet18(weights=None)
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
backbone.maxpool, backbone.fc = nn.Identity(), nn.Identity()

byol = BYOL(backbone).to(device)
optimizer = optim.Adam(byol.parameters(), lr=lr)

# === ENTRAÎNEMENT BYOL ===
print("🔧 Préentraînement BYOL...")
for epoch in range(pretrain_epochs):
    total_loss = 0
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        loss = byol(x1, x2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        byol.update_moving_average()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(loader):.4f}")

# === LINEAR PROBING ===
print("\n🔍 Évaluation Linear Probing...")
transform_eval = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])
train_eval = datasets.CIFAR10(root="./data", train=True, transform=transform_eval)
test_eval = datasets.CIFAR10(root="./data", train=False, transform=transform_eval)
train_loader = DataLoader(train_eval, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_eval, batch_size=batch_size)

class LinearProbe(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        for p in encoder.parameters(): p.requires_grad = False
        self.encoder = encoder
        self.fc = nn.Linear(512, 10)

    def forward(self, x): return self.fc(self.encoder(x))

probe = LinearProbe(byol.online_encoder[0]).to(device)
optimizer = optim.Adam(probe.fc.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(finetune_epochs):
    correct, total, total_loss = 0, 0, 0
    probe.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = probe(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    print(f"[FineTune {epoch+1}] Loss: {total_loss:.4f}, Acc: {correct/total:.4f}")

# === TEST FINAL ===
probe.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = probe(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"\n🎯 Accuracy final avec ondelette [{wavelet_type}] : {correct/total:.4f}")
