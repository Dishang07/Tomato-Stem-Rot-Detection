import os
import torch
from torchvision import transforms
from PIL import Image

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_class, model_path, num_classes=2):
    model = model_class(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=get_device()))
    model.to(get_device())
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(get_device())


def predict_image(model, image_tensor, class_names=["Healthy", "Diseased"]):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]
