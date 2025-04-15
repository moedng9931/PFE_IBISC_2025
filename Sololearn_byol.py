import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import copy

# ------------------ CONFIG ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.001
pretrain_epochs = 30
finetune_epochs = 30
ema_m = 0.99

# ------------------ AUGMENTATION ------------------
def get_simclr_augmentation():
    return transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# ------------------ DATASET WRAPPER ------------------
class BYOLDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return self.transform(img), self.transform(img)

# ------------------ MODEL ------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, backbone, hidden_dim=4096, out_dim=256, m=0.996):
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

# ------------------ PRETRAIN BYOL ------------------
transform = get_simclr_augmentation()
dataset = datasets.CIFAR10(root='./data', train=True, download=True)
pretrain_loader = DataLoader(BYOLDataset(dataset, transform), batch_size=batch_size, shuffle=True)



backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
backbone.maxpool = nn.Identity()  # important pour CIFAR-10
backbone.fc = nn.Identity()

backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
backbone.fc = nn.Identity()

byol_model = BYOL(backbone, m=ema_m).to(device)
optimizer = optim.Adam(byol_model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)

print("\n==> Pretraining BYOL...")
for epoch in range(pretrain_epochs):
    byol_model.train()
    total_loss = 0
    for x1, x2 in pretrain_loader:
        x1, x2 = x1.to(device), x2.to(device)
        loss = byol_model(x1, x2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        byol_model.update_moving_average()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch [{epoch+1}/{pretrain_epochs}], BYOL Loss: {total_loss/len(pretrain_loader):.4f}")

# ------------------ FINE-TUNING ------------------
print("\n==> Fine-tuning classifier...")
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

classifier = copy.deepcopy(byol_model.online_encoder[0])
classifier.fc = nn.Linear(512, 10)
classifier = classifier.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def train_model(model, trainloader, criterion, optimizer, scheduler, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {correct/total:.4f}")
    return model

def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"\nPrécision sur le jeu de test : {correct / total:.4f}")

classifier = train_model(classifier, trainloader, criterion, optimizer, scheduler, finetune_epochs)
evaluate_model(classifier, testloader)
