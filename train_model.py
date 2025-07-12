import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Load dataset and preprocess
metadata = pd.read_csv('data/HAM10000_metadata.csv')
image_data = []
labels = []

label_map = {label: i for i, label in enumerate(metadata['dx'].unique())}

image_dir_1 = 'data/HAM10000_images_part_1/'
image_dir_2 = 'data/HAM10000_images_part_2/'

for index, row in metadata.iterrows():
    image_id = row['image_id']
    diagnosis = row['dx']
    label = label_map[diagnosis]

    image_path = os.path.join(image_dir_1, image_id + '.jpg')
    if not os.path.exists(image_path):
        image_path = os.path.join(image_dir_2, image_id + '.jpg')

    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
        image_data.append(img)
        labels.append(label)

X = np.array(image_data) / 255.0  # Normalize images
y = to_categorical(labels)  # One-hot encode labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('model_skin_cancer.h5')
