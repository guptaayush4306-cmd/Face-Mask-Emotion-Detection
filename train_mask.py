import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

print("MASK TRAINING STARTED")

data = []
labels = []

path = "dataset/mask"
categories = ["with_mask", "without_mask"]

# LOAD IMAGES
for i, category in enumerate(categories):
    folder = os.path.join(path, category)

    print("Loading from:", folder)

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (100, 100))
        data.append(image)
        labels.append(i)

print("Total images loaded:", len(data))

# STOP if no data
if len(data) == 0:
    print("❌ No images found! Check dataset.")
    exit()

# PREPROCESS
data = np.array(data) / 255.0
labels = to_categorical(labels, 2)

# MODEL
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TRAIN
model.fit(data, labels, epochs=10, batch_size=32)

# SAVE
model.save("mask_model.h5")

print("✅ MASK MODEL SAVED!")