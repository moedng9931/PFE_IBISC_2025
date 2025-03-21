import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ------------------ CONFIGURATION ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64  # Assurez-vous que batch_size > 1
learning_rate = 0.001
pretrain_epochs = 10
finetune_epochs = 10

# ------------------ MLP + BYOL MODULES ------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        if x.shape[0] == 1:  # Éviter BatchNorm si batch_size == 1
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
        else:
            x = self.linear1(x)
            x = self.batchnorm(x)
            x = self.relu(x)
            x = self.linear2(x)
        return x

class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=256, hidden_dim=4096, moving_average_decay=0.99):
        super().__init__()
        self.online_encoder = nn.Sequential(base_encoder, MLPHead(512, hidden_dim, projection_dim))
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = MLPHead(projection_dim, hidden_dim, projection_dim)
        self.m = moving_average_decay
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def update_moving_average(self):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.m * target_params.data + (1 - self.m) * online_params.data

    def forward(self, x1, x2):
        proj1 = self.predictor(self.online_encoder(x1))
        with torch.no_grad():
            proj2 = self.target_encoder(x2)
        proj1 = nn.functional.normalize(proj1, dim=-1)
        proj2 = nn.functional.normalize(proj2, dim=-1)
        return 2 - 2 * (proj1 * proj2).sum(dim=-1).mean()

# ------------------ DATA AUGMENTATION ------------------
augment = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def tensor_to_pil(tensor):
    """ Convertit un tensor en image PIL """
    img = tensor.permute(1, 2, 0).numpy()  # Réorganiser les dimensions (H, W, C)
    img = (img * 255).astype(np.uint8)  # Convertir en uint8
    return Image.fromarray(img)

byol_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
byol_loader = DataLoader(byol_dataset, batch_size=batch_size, shuffle=True)

# ------------------ BYOL PRETRAINING ------------------
encoder = models.resnet18(weights=None)
encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
encoder.fc = nn.Identity()  # remove classification layer

byol_model = BYOL(encoder).to(device)
optimizer = optim.Adam(byol_model.parameters(), lr=learning_rate)

print("🔧 BYOL Pretraining ...")
for epoch in range(pretrain_epochs):
    byol_model.train()
    total_loss = 0.0
    for (x, _) in byol_loader:
        x1 = augment(tensor_to_pil(x[0])).unsqueeze(0).to(device)  # Convertir en PIL + Augmentation
        x2 = augment(tensor_to_pil(x[0])).unsqueeze(0).to(device)

        loss = byol_model(x1, x2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        byol_model.update_moving_average()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{pretrain_epochs}], BYOL Loss: {total_loss/len(byol_loader):.4f}")

# ------------------ FINE-TUNING ------------------
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

# Fine-tune ResNet18
finetune_model = copy.deepcopy(byol_model.online_encoder[0])
finetune_model.fc = nn.Linear(512, 10)
finetune_model = finetune_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(finetune_model.parameters(), lr=learning_rate)

def train_model(model, trainloader, criterion, optimizer, num_epochs=10):
    train_loss_history = []
    train_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
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

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_history)
    plt.title("Train Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_history)
    plt.title("Train Accuracy")
    plt.show()

    return model

def evaluate_model(model, testloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"✅ Test Accuracy: {accuracy:.4f}")

print("📦 Fine-tuning classifier ...")
finetune_model = train_model(finetune_model, trainloader, criterion, optimizer, finetune_epochs)
evaluate_model(finetune_model, testloader)

