# imports
import numpy as np
import cv2 as cv
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import information
import random

# information calling
NUM_EPOCHS = information.NUM_EPOCHS
BATCH_SIZE = information.BATCH_SIZE
INPUT_SHAPE = information.INPUT_SHAPE
TRIAL = information.TRIAL
MODEL_SAVE_PATH = information.MODEL_SAVE_PATH
HISTORY_SAVE_PATH = information.HISTORY_SAVE_PATH

"""#Data"""

from PIL import Image
def load_hazy_dataset(directory, image_size=(256, 256)):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or
                                                        f.endswith('.jpg') or
                                                        f.endswith('.jpeg')])
    images = []

    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = image.resize(image_size)
        images.append(np.array(image) / 255.0)
        # print("Loaded image ", i)

    return np.array(images)

from PIL import Image
def load_clear_dataset(directory, image_size=(256, 256)):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or
                                                        f.endswith('.jpg') or
                                                        f.endswith('.jpeg')])
    images = []

    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = image.resize(image_size)
        images.append(np.array(image) / 255.0)
        # print("Loaded image ", i)

    return np.array(images)

hazy_folder_path = 'SMOKE/train/hazy/'
hazy_images = load_hazy_dataset(hazy_folder_path)

clear_folder_path = 'SMOKE/train/clean/'
clear_images = load_clear_dataset(clear_folder_path)

# data augmentation
def random_flip(image, label):
    if random.random() > 0.5:
        image = np.fliplr(image)
        label = np.fliplr(label)
    if random.random() > 0.5:
        image = np.flipud(image)
        label = np.flipud(label)
    return image, label

def random_rotate(image, label):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-180, 180)

    M = cv.getRotationMatrix2D(center, angle, 1.0)

    image = cv.warpAffine(image, M, (w, h), flags=cv.INTER_LINEAR)
    label = cv.warpAffine(label, M, (w, h), flags=cv.INTER_LINEAR)

    return image, label

def augment_image_and_label(image, label):
    image, label = random_flip(image, label)
    image, label = random_rotate(image, label)
    return image, label

def augment_dataset(images, labels):
    augmented_images = []
    augmented_labels = []

    multiple = 7
    for image, label in zip(images, labels):
        for i in range(multiple):            
            augmented_image, augmented_label = augment_image_and_label(image, label)
            augmented_images.append(augmented_image)
            augmented_labels.append(augmented_label)
            
    return np.array(augmented_images), np.array(augmented_labels)

augmented_train_image, augmented_train_label = augment_dataset(hazy_images, clear_images)
if augmented_train_label.ndim == 3:
    augmented_train_label = np.expand_dims(augmented_train_label, axis=-1)

print (augmented_train_image.shape, augmented_train_label.shape)

combined_train_image = np.concatenate([hazy_images, augmented_train_image], axis=0)
combined_train_label = np.concatenate([clear_images, augmented_train_label], axis=0)

# Considering the size of the dataset, data augmentation is probably unneccessary
# and would take a lot of computing resources

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(combined_train_image, combined_train_label, test_size=0.2, random_state=42, shuffle=True)

"""#Model Training"""
from tensorflow.keras import layers, models

def build_dehazing_model(input_shape=INPUT_SHAPE):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder - Reduced number of filters
    # Initial convolution
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)
    skip1 = x  # Skip connection 1

    # Encoder block 1
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    skip2 = x  # Skip connection 2

    # Encoder block 2 - Reduced filters
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Bridge - Single dilated convolution
    x = layers.Conv2D(128, (3, 3), padding='same', dilation_rate=2)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    # Decoder block 1
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Dropout(0.5)(x)

    # Decoder block 2
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, skip1])

    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="LightDehazer")
    return model
# Build and compile model
model = build_dehazing_model()
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Update the training
history = model.fit(
    X_train,
    y_train,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[
        early_stopping_callback,
    ]
)

model.save(MODEL_SAVE_PATH)

import pickle
with open(HISTORY_SAVE_PATH, 'wb') as file:
  pickle.dump(history.history, file)