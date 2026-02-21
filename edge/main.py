# disabling tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# imports
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
from PIL import Image
import information
import random

# Getting the information from information.py module
TRIAL_NUMBER = information.TRIAL_NUMBER
MODEL_SAVE_PATH = information.MODEL_SAVE_PATH
HISTORY_SAVE_PATH = information.HISTORY_SAVE_PATH
EPOCH = information.EPOCH
BATCH = information.BATCH

# get data from /BIPEDv2
dataset_path = 'BIPEDv2/BIPED/edges'

train_image_path = dataset_path + '/imgs/train/rgbr/real/'
train_label_path = dataset_path + '/edge_maps/train/rgbr/real/'

test_image_path = dataset_path +  '/imgs/test/rgbr/'
test_label_path = dataset_path +  '/edge_maps/test/rgbr/'

def load_dataset(directory, image_size=(256, 256), printName=False):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or 
                                                        f.endswith('.jpg') or 
                                                        f.endswith('.jpeg')])
    length = len(image_files)
    images = np.empty((length, image_size[0], image_size[1], 1 if 'edge_maps'  in directory else 3), dtype='float32')

    for i, filename in enumerate(image_files):
        if printName: print(filename)
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        if 'edge_maps' in directory:
            image.convert('L')
        image = image.resize(image_size)

        if 'edge_maps' in directory:
            images[i, :, :, 0] = np.array(image) / 255.0
        else:
            images[i] = np.array(image) / 255.0

    return images

train_image = load_dataset(train_image_path)
train_label = load_dataset(train_label_path)
test_image = load_dataset(test_image_path)
test_label = load_dataset(test_label_path)

# preprocess data
train_image = train_image.astype('float32') 
train_label = train_label.astype('float32')

test_image = test_image.astype('float32') 
test_label = test_label.astype('float32')

# Data augmentation functions
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

    multiple = 5
    for image, label in zip(images, labels):
        for i in range(multiple):            
            augmented_image, augmented_label = augment_image_and_label(image, label)
            augmented_images.append(augmented_image)
            augmented_labels.append(augmented_label)
            
    return np.array(augmented_images), np.array(augmented_labels)

# Apply augmentation to the training data
augmented_train_image, augmented_train_label = augment_dataset(train_image, train_label)
if augmented_train_label.ndim == 3:
    augmented_train_label = np.expand_dims(augmented_train_label, axis=-1)

print (augmented_train_image.shape, augmented_train_label.shape)

# Combine original and augmented data
combined_train_image = np.concatenate([train_image, augmented_train_image], axis=0)
combined_train_label = np.concatenate([train_label, augmented_train_label], axis=0)

def visualize_image_and_label(image, label, start, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        # Plot the image
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(image[start + i])
        plt.title(f"Image {i + 1}")
        plt.axis('off')

        # Plot the corresponding label
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(label[start + i], cmap='gray')
        plt.title(f"Label {i + 1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_image_and_label(combined_train_image, combined_train_label, start=200, num_samples=10)

# callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# initialize model
def cnn_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Modify encoder to store both pre and post stride outputs
    def encoder_block(x, filters, strides=(2, 2), kernel_size=(3, 3), padding='same', activation='relu'):
        # First conv without stride to get skip connection
        skip = layers.SeparableConv2D(filters, kernel_size, padding=padding)(x)
        skip = layers.BatchNormalization()(skip)
        skip = layers.Activation(activation)(skip)
        
        # Second conv with stride for downsampling
        x = layers.SeparableConv2D(filters, kernel_size, strides=strides, padding=padding)(skip)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        return x, skip

    # Upsampling (unchanged)
    def decoder_block(x, skip_features, filters, kernel_size=(3, 3), padding='same', activation='relu'):
        x = layers.UpSampling2D(size=(2, 2))(x)
        # x = layers.SeparableConv2D(filters, kernel_size, padding=padding)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation(activation)(x)
        x = layers.concatenate([x, skip_features])
        x = layers.SeparableConv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x

    x = inputs
    x, s1 = encoder_block(x, 32, strides=(2, 2))    # 128x128
    x, s2 = encoder_block(x, 64, strides=(2, 2))    # 64x64
    # x, s3 = encoder_block(x, 128, strides=(2, 2))   # 32x32

    bottleneck = x  # Already at 32x32

    # d1 = decoder_block(bottleneck, s3, 128)   # 64x64
    d2 = decoder_block(bottleneck, s2, 64)      # 128x128
    d3 = decoder_block(d2, s1, 32)              # 256x256

    outputs = layers.SeparableConv2D(1, (1, 1), activation='sigmoid', padding='same')(d3)
    
    model = models.Model(inputs, outputs, name='EdgeDetectionModel')
    return model

# model compile and fit
model = cnn_model(input_shape=(256, 256, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    combined_train_image, 
    combined_train_label, 
    epochs=EPOCH,
    batch_size=BATCH, 
    validation_split = 0.2,
    validation_batch_size=BATCH, 
    callbacks=[early_stopping_callback, checkpoint_callback]
    )

# save model and history
model.save(MODEL_SAVE_PATH)

import pickle
with open(HISTORY_SAVE_PATH, 'wb') as file:
  pickle.dump(history.history, file)

test_loss, test_accuracy = model.evaluate(test_image, test_label)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Plotting the history
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("training and validation accuracy")
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)
plt.show()