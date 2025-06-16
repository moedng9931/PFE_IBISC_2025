import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

# ============ CONFIGURATION ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
learning_rate = 0.001
num_epochs = 100
label_fraction = 0.01  # 👈 change ici : 0.01 (1%), 0.2 (20%), 0.5 (50%)

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# ============ TRANSFORMATIONS ============
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ============ FUNCTION TO REDUCE DATASET ============
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

# ============ DATASETS ============
full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
subset_trainset = get_subset_per_class(full_trainset, label_fraction)
trainloader = DataLoader(subset_trainset, batch_size=batch_size, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# ============ MODEL ============
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# ============ OPTIMIZATION ============
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# ============ TRAINING ============
for epoch in range(num_epochs):
    model.train()
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

    acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss:.4f} - Accuracy: {acc:.4f}")
    scheduler.step()

# ============ TEST ============
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

print(f"\n🎯 Test accuracy with {int(label_fraction*100)}% labels: {100 * correct / total:.2f}%")
