# train.py
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict

# Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset and save the model.")
    
    parser.add_argument('data_dir', type=str, help='Directory of training and validation data')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16, vgg13, etc.)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=2048, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    return parser.parse_args()

# Set up data transformations and loaders
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Define transformations
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    
    return trainloader, validloader, train_dataset.class_to_idx

# Build model
def build_model(arch, hidden_units):
    model = getattr(models, arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

# Training loop
def train(model, trainloader, validloader, criterion, optimizer, device, epochs):
    model.to(device)
    steps = 0
    print_every = 5
    
    for epoch in range(epochs):
        running_loss = 0
        
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

# Save checkpoint
def save_checkpoint(model, optimizer, epochs, save_dir, class_to_idx):
    checkpoint = {
        'pretrained_model': 'vgg16',
        'input_size': 25088,
        'output_size': 102,
        'classifier': model.classifier,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')
    print(f"Checkpoint saved to {save_dir}/checkpoint.pth")

# Main function
def main():
    args = get_args()
    
    # Load data
    trainloader, validloader, class_to_idx = load_data(args.data_dir)
    
    # Build model
    model = build_model(args.arch, args.hidden_units)
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Determine device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Train model
    train(model, trainloader, validloader, criterion, optimizer, device, args.epochs)
    
    # Save checkpoint
    save_checkpoint(model, optimizer, args.epochs, args.save_dir, class_to_idx)

if __name__ == '__main__':
    main()
