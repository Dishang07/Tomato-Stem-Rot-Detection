# scripts/metrics.py

import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
data_dir = "../data"
model_dir = "../models"
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset and loader
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
class_names = dataset.classes  # ['diseased', 'healthy']

# Helper function to evaluate each model
def evaluate_model(model, model_name):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n📊 Results for {model_name}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Save confusion matrix heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

# Load model based on name
def load_model(model_name):
    if model_name == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model_path = os.path.join(model_dir, "resnet50_tomato.pth")
    elif model_name == "AlexNet":
        model = models.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 2)
        model_path = os.path.join(model_dir, "alexnet.pth")
    elif model_name == "GoogLeNet":
        model = models.googlenet(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model_path = os.path.join(model_dir, "googlenet.pth")
    elif model_name == "VGG16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 2)
        model_path = os.path.join(model_dir, "vgg16.pth")
    else:
        raise ValueError("Unsupported model!")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

# Evaluate all models
model_names = ["ResNet50", "AlexNet", "GoogLeNet", "VGG16"]

for name in model_names:
    try:
        model = load_model(name)
        evaluate_model(model, name)
    except Exception as e:
        print(f"❌ Failed to evaluate {name}: {e}")
