import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
DATA_DIR = "../../data/GTSRB"
MODEL_PATH = "model.h5"

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(43, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model.fit(train_data, epochs=10, validation_data=val_data)
model.save(MODEL_PATH)
