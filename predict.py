# predict.py

import torch
from torchvision import transforms
from PIL import Image
import os

def cryptic_inf_f(model, list_of_img_paths, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    images = []
    for img_path in list_of_img_paths:
        img = Image.open(img_path)
        img = transform(img)
        images.append(img)
        
    images = torch.stack(images).to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
    
    labels = ["smiling" if label == 1 else "not smiling" for label in predicted]
    return labels
