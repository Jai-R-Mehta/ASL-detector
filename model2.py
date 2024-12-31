import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------
# Data Preparation
# ----------------------
class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.label_map = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
            'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'O': 13, 'P': 14,
            'Q': 15, 'R': 16, 'S': 17, 'T': 18, 'U': 19, 'V': 20, 'W': 21,
            'X': 22, 'Y': 23
        }

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
                            continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# ----------------------
# Define CNN Model
# ----------------------
class ASL_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 24)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------
# Training and Evaluation Functions
# ----------------------
def train_model(model, train_loader, val_loader, loss_function, optimizer, device, epochs=15, patience=3):
    train_losses, val_losses = [], []
    best_val_loss = 0.0369
    early_stop_counter = 0

    for epoch in range(epochs):
        # Training phase
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

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += loss_function(outputs, labels).item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

        # Early stopping logic
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_asl_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs. Best Validation Loss: {best_val_loss:.4f}")
                break

    return train_losses, val_losses

def evaluate_per_class(model, loader, device, label_map):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    reverse_label_map = {v: k for k, v in label_map.items()}
    print(classification_report(y_true, y_pred, target_names=[reverse_label_map[i] for i in range(len(label_map))]))

# ----------------------
# Main
# ----------------------
data_dir = "SignData"
batch_size = 64
epochs = 15
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),  # Scale down to 64x64
    transforms.ToTensor(),       # Convert to tensor
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale values
])

dataset = ASLDataset(data_dir, transform=transform)
labels = np.array(dataset.labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
for train_indices, val_indices in kfold.split(np.zeros(len(labels)), labels):
    print(f"\nFold {fold}")
    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)



    model = ASL_CNN().to(device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_losses, val_losses = train_model(model, train_loader, val_loader, loss_function, optimizer, device, epochs)

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Fold {fold} - Training vs. Validation Loss")
    plt.show()

    evaluate_per_class(model, val_loader, device, dataset.label_map)
    fold += 1
torch.save(model.state_dict(), "asl_model2.pth")