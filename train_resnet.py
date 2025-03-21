import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import copy
import time
import matplotlib.pyplot as plt

# Définir le dispositif (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamètres
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Transformations et augmentation des données
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Chargement des données CIFAR-10
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Charger le modèle ResNet18 pré-entraîné
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Adapter la première couche (évite une perte excessive de l'information)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

# Adapter la couche de sortie pour 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

# Envoyer le modèle sur le GPU si disponible
model = model.to(device)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Fonction d'entraînement avec suivi des métriques
def train_model(model, trainloader, criterion, optimizer, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_history = []
    train_acc_history = []

    for epoch in range(num_epochs):
        model.train()  # Mode entraînement
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Réinitialisation des gradients

            # Passage avant
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Passage arrière et optimisation
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

        # Sauvegarde du modèle avec la meilleure précision
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print('Entraînement terminé!')
    model.load_state_dict(best_model_wts)

    # Tracer les courbes d'entraînement
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_history, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_history, label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()

    plt.show()

    return model

# Entraîner le modèle
model = train_model(model, trainloader, criterion, optimizer, num_epochs)

# Fonction d'évaluation
def evaluate_model(model, testloader):
    model.eval()  # Mode évaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Précision sur le jeu de test : {accuracy:.4f}')

# Évaluer le modèle sur le jeu de test
evaluate_model(model, testloader)
