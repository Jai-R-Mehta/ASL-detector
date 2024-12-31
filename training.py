import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import pickle

random.seed(42)

# Dataset Preparation
class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Label Map
        self.label_map = {
            'A': 0, 'B': 1, 'C': 2, 
            'D': 3, 'E': 4, 'F': 5, 
            'G': 6, 'H': 7, 'I': 8, 
            'K': 9, 'L': 10, 'M': 11, 
            'N': 12, 'O': 13, 'P': 14, 
            'Q': 15, 'R': 16, 'S': 17, 
            'T': 18, 'U': 19, 'V': 20, 
            'W': 21, 'X': 22, 'Y': 23
        }

        # Load Images and Labels
        for label in os.listdir(root_dir):
            if label in self.label_map:  
                label_dir = os.path.join(root_dir, label)
                if os.path.isdir(label_dir):
                    for img_file in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_file)
                        try:
                            with Image.open(img_path) as img:
                                img.verify()
                            self.data.append(img_path)
                            self.labels.append(self.label_map[label])
                        except (IOError, SyntaxError):
                            print(f"Skipping non-image file: {img_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# CNN Model Definition
class ASL_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 24)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training Function
def train_model(model, train_loader, val_loader, loss_function, optimizer, device, epochs=15):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += loss_function(outputs, labels).item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")
    return train_losses, val_losses
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    epochs = 15
    learning_rate = 0.001
    data_dir = "SignData"

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Prepare Dataset and Loaders
    dataset = ASLDataset(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASL_CNN().to(device)
    model.apply(lambda m: nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Conv2d) else None)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Model
    train_losses, val_losses = train_model(model, train_loader, val_loader, loss_function, optimizer, device, epochs)

    # Save Model
    with open("asl_model.pkl", "wb") as f:
        pickle.dump(model.state_dict(), f)

    print("Model saved as asl_model.pkl")
