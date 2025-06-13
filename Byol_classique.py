import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import copy

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr = 1e-3
pretrain_epochs = 100
finetune_epochs = 50

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

# --- LOAD CIFAR10 ---
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

# --- LINEAR PROBING ---
linear_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.CIFAR10(root='./data', train=True, transform=linear_transform)
testset = datasets.CIFAR10(root='./data', train=False, transform=linear_transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size)

class LinearClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        for p in encoder.parameters():
            p.requires_grad = False
        self.encoder = encoder
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc(self.encoder(x))

linear_model = LinearClassifier(byol_model.online_encoder[0]).to(device)
optimizer = optim.Adam(linear_model.fc.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(finetune_epochs):
    linear_model.train()
    correct, total, total_loss = 0, 0, 0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        out = linear_model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    print(f"[Linear Epoch {epoch+1}] Loss: {total_loss:.4f}, Accuracy: {correct/total:.4f}")

# --- TEST ---
linear_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        pred = linear_model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"\n🧪 Final Test Accuracy: {correct / total:.4f}")
