# predict.py
import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np

# Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="Predict flower class and probability from an image.")
    
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()

# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = getattr(models, checkpoint['pretrained_model'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Process image
def process_image(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(img)

# Predict
def predict(image_path, model, top_k, device):
    model.to(device)
    model.eval()
    
    image = process_image(image_path).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
    
    top_p, top_class = ps.topk(top_k, dim=1)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i.item()] for i in top_class[0]]
    
    return top_p[0].tolist(), top_classes

# Load category names
def load_category_names(category_names):
    with open(category_names, 'r') as f:
        return json.load(f)

# Main function
def main():
    args = get_args()
    
    # Load model
    model = load_checkpoint(args.checkpoint)
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Make prediction
    probs, classes = predict(args.input, model, args.top_k, device)
    
    # Load category names if provided
    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
        class_names = [cat_to_name[c] for c in classes]
    else:
        class_names = classes
    
    # Print predictions
    for prob, class_name in zip(probs, class_names):
        print(f"Class: {class_name}, Probability: {prob:.4f}")

if __name__ == '__main__':
    main()
