import cv2
import numpy as np

def preprocess(img):
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    return img
