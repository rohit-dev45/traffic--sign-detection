import cv2
import numpy as np
from tensorflow.keras.models import load_model
from .preprocess import preprocess

model = load_model('../model/model.h5')
with open('../model/labels.txt', 'r') as f:
    labels = f.read().splitlines()

def detect(frame):
    img = preprocess(frame)
    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = np.max(preds)
    return labels[class_id], confidence
