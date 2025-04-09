import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

def load_data(data_dir):
    images = []
    labels = []
    for label in ["yes", "no"]:
        path = os.path.join(data_dir, label)
        class_num = 1 if label == "yes" else 0
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))
            images.append(image)
            labels.append(class_num)
    images = np.array(images).reshape(-1, 128, 128, 1) / 255.0
    labels = np.array(labels)
    return images, labels

data_dir = "D:\\brain_tumor_detection\\augmented_data"
images, labels = load_data(data_dir)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

input_shape = (128, 128, 1)
model_input = Input(shape=input_shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(model_input)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=model_input, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Less aggressive early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history=model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), callbacks=[early_stop])

# Save
model.save("D:\\brain_tumor_detection\\brain_tumor_detection_model.h5")

import json

# Save the training history
import json
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

