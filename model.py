import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import random

random.seed(42)
# Data Preparation
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

        print("Label Map:", self.label_map)  # Debug: Print label map

        for label in os.listdir(root_dir):
            if label in self.label_map:  
                label_dir = os.path.join(root_dir, label)
                if os.path.isdir(label_dir):
                    for img_file in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_file)
                        try:
                            # Check if the file can be opened as an image
                            with Image.open(img_path) as img:
                                img.verify()  # Verify if the file is an image
                            self.data.append(img_path)
                            self.labels.append(self.label_map[label])
                        except (IOError, SyntaxError):
                            print(f"Skipping non-image file: {img_path}")  # Debug: Print skipped files

        print(f"Total samples loaded: {len(self.data)}")  # Debug: Print total samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Debug: Print a few samples
        if idx % 10000 ==0:
            print(f"Sample {idx}: Image Path: {img_path}, Label: {label}")
        
        return image, label

# Define CNN Model
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
            nn.Linear(128 * 6 * 6, 256),  # Adjust based on input resolution
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 24)  # 24 classes (excluding J and Z)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training Function to train with the train_loader and validate with the val_loader
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

# Evaluates model with the test set
def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Directory 
data_dir = "SignData"


# Hyperparameters
batch_size = 32
epochs = 15
learning_rate = 0.001

# Transformation; Resizes and Normalizes the images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),  # Scale down to 64x64
    transforms.ToTensor(),       # Convert to tensor
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale values
])

# Dataset and DataLoaders
dataset = ASLDataset(data_dir, transform=transform)
def visualize_transformed_images(dataset, num_images=5):
    label_seen = set()
    plt.figure(figsize=(10, 5))
    images_shown = 0

    for i in range(len(dataset)):
        if images_shown >= num_images:
            break
        image, label = dataset[i]  # Get the transformed image and label
        if label not in label_seen:
            label_seen.add(label)
            image = image.squeeze(0).numpy()  # Remove channel dimension and convert to numpy array
            plt.subplot(1, num_images, images_shown + 1)
            plt.imshow(image, cmap="gray")
            plt.title(f"Label: {label}")
            plt.axis("off")
            images_shown += 1

    plt.tight_layout()
    plt.show()

# Display a few images
visualize_transformed_images(dataset, num_images=5)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")  # Debug: Split sizes

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(val_data, batch_size=batch_size)  # Using validation set as test set for simplicity


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Model, Loss, Optimizer
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = ASL_CNN().to(device)
model.apply(init_weights) 
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Model
train_losses, val_losses = train_model(model, train_loader, val_loader, loss_function, optimizer, device, epochs)

# Plot Training and Validation Loss
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs. Validation Loss")
plt.show()

# Evaluate Model
evaluate_model(model, test_loader, device)



# Saves model for use by detector.py
torch.save(model.state_dict(), "asl_model.pth")