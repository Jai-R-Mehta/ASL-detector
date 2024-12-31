import cv2
import torch
import numpy as np
import torch.nn as nn
import math
from cvzone.HandTrackingModule import HandDetector
from torchvision import transforms
from PIL import Image

# Define the ASL_CNN model (same as in model.py)
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


# Initialize Camera and Hand Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load your trained PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASL_CNN()
model.load_state_dict(torch.load("asl_model.pth", map_location=device))
model.to(device)
model.eval()

# Parameters
imgSize = 128  # Should match training input size
offset = 20

# Label Mapping
label_map = {
    0: 'A', 1: 'B', 2: 'C', 
    3: 'D', 4: 'E', 5: 'F', 
    6: 'G', 7: 'H', 8: 'I', 
    9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 
    15: 'Q', 16: 'R', 17: 'S', 
    18: 'T', 19: 'U', 20: 'V', 
    21: 'W', 22: 'X', 23: 'Y'
    }

# Transformation (should match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),  # Ensure grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Main Loop
while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Prepare input image for model
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.size != 0:  # Ensure the crop is valid
            aspect_ratio = h / w
            if aspect_ratio > 1:
                k = imgSize / h
                wCalc = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCalc, imgSize))
                wGap = math.ceil((imgSize - wCalc) / 2)
                imgWhite[:, wGap:wCalc + wGap] = imgResize
            else:
                k = imgSize / w
                hCalc = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCalc))
                hGap = math.ceil((imgSize - hCalc) / 2)
                imgWhite[hGap:hCalc + hGap, :] = imgResize

            # Convert to PIL image and apply transformations
            imgPIL = Image.fromarray(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB))
            input_tensor = transform(imgPIL).unsqueeze(0).to(device)

            # Perform prediction
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = label_map[int(predicted)]

            # Display predictions
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x + w + offset, y - offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, predicted_label, (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # Show the final image
    cv2.imshow("Image", imgOutput)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
