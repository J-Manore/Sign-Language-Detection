import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# Constants
OFFSET = 20
IMG_SIZE = 300
FOLDER = "Data/Thank You"
LABELS_FILE = "Codes/Model/labels.txt"
MODEL_FILE = "Codes/Model/keras_model.h5"
LABELS = ["I Love You", "Thank You"]  # Update dynamically if needed

# Initialize camera and modules
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(MODEL_FILE, LABELS_FILE)

counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        imgCrop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]

        imgCropShape = imgCrop.shape

        # Ensure imgCrop is not empty before resizing
        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = IMG_SIZE / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                wGap = math.ceil((IMG_SIZE - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = IMG_SIZE / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                hGap = math.ceil((IMG_SIZE - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            cv2.putText(imgOutput, LABELS[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), (255, 0, 255), 4)

            # Display the cropped hand image only if it has valid dimensions
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        
    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    elif key == ord("s"):
        counter += 1
        cv2.imwrite(f'{FOLDER}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
