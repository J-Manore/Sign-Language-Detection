from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Constants
OFFSET = 20
IMG_SIZE = 300
LABELS_FILE = "Codes/Model/labels.txt"
MODEL_FILE = "Codes/Model/keras_model.h5"
LABELS = ["I Love You", "Thank You"]

app = Flask(__name__)
detector = HandDetector(maxHands=1)
classifier = Classifier(MODEL_FILE, LABELS_FILE)

@app.route('/')
def home():
    return send_from_directory('', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Check if image is captured correctly
        if img is None:
            return jsonify({'gesture': 'Failed to read image'})

        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
            imgCrop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
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
                print(f"Predicted Gesture: {LABELS[index]}")  # Debug output
                return jsonify({'gesture': LABELS[index]})

        return jsonify({'gesture': 'No hand detected'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
