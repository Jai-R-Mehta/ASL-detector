import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Initialize the HandDetector from cvzone library to detect one hand at a time
detector = HandDetector(maxHands=1)

# Define constants for image size, cropping offset, and counter for saved images
imgSize = 300  # Desired size for processed images
offset = 20  # Margin around the detected hand
counter = 0  # Counter to keep track of saved images

# Define the folder path where the images will be stored
folder = "SignData/X"

while True:
    # Capture a frame from the webcam
    success, img = cap.read()

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    if hands:
        # Extract bounding box information of the first detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']  # x, y, width, and height of the bounding box

        # Create a white canvas of the desired size for normalization
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the Hightlight box around the detected hand
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Get the shape of the cropped image
        imgCropShape = imgCrop.shape

        # Calculate the aspect ratio of the cropped hand region
        aspect_ratio = h / w

        # Adjust the cropped image to fit into a square canvas
        if aspect_ratio > 1:
            # If height > width, resize the height to the desired size and calculate the corresponding width
            k = imgSize / h
            wCalc = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCalc, imgSize))  # Resize keeping the aspect ratio
            imgResizeShape = imgResize.shape

            # Calculate horizontal padding to center the image
            wGap = math.ceil((imgSize - wCalc) / 2)
            imgWhite[0:imgResizeShape[0], wGap:wCalc + wGap] = imgResize  # Place the resized image on the white canvas
        else:
            # If width > height, resize the width to the desired size and calculate the corresponding height
            k = imgSize / w
            hCalc = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCalc))  # Resize keeping the aspect ratio
            imgResizeShape = imgResize.shape

            # Calculate vertical padding to center the image
            hGap = math.ceil((imgSize - hCalc) / 2)
            imgWhite[hGap:hCalc + hGap, 0:imgResizeShape[1]] = imgResize  # Place the resized image on the white canvas

        # Display the cropped image and the normalized image
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    # Display the original image with hand detection
    cv2.imshow("Image", img)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If 's' is pressed, save the normalized image to the specified folder
    if key == ord("s"):
        counter += 1  
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # Save the image with a unique name
        print(counter)  
