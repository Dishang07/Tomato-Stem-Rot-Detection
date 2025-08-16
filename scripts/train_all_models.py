import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import googlenet, GoogLeNet_Weights

# Configs
data_dir = "../data"
model_dir = "../models"
batch_size = 8
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and Loader
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss
criterion = nn.CrossEntropyLoss()

def train_model(model, model_name):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n🔧 Training {model_name}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(loader):.4f}")

    # Save model
    save_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved {model_name} to {save_path}")

# AlexNet
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 2)
train_model(alexnet, "alexnet")

# GoogLeNet
weights = GoogLeNet_Weights.DEFAULT
googlenet = googlenet(weights=weights, aux_logits=True)
googlenet.fc = nn.Linear(googlenet.fc.in_features, 2)

# Disable auxiliary classifiers if not needed
googlenet.aux_logits = False
googlenet.aux1 = None
googlenet.aux2 = None

train_model(googlenet, "googlenet")

# VGG16
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 2)
train_model(vgg16, "vgg16")
