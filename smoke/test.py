# removing tf warnings
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# imports
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
from PIL import Image
import information
import random

model_path = information.MODEL_SAVE_PATH
model = load_model(model_path)

test_dir = 'SMOKE/test/hazy'

def extract_number(filename):
    match = re.search(r'\d+', filename)

    if match:
        return int(match.group())
    else: return 0

# Preprocess prediction image
images_files = sorted([f for f in os.listdir(test_dir) 
                       if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')], key=extract_number)
print(images_files)

image_size = (256, 256)
num_samples = len(images_files)
i = 0

def get_prediction(filename):
    image_path = os.path.join(test_dir, filename)

    img = Image.open(image_path)
    img = img.resize(image_size)

    image = np.array(img) / 255.0
    image = np.expand_dims(image, axis=0) 
    
    print("Image shape:", image.shape)

    prediction = model.predict(image)
    print("Prediction shape:", prediction.shape)
        
    image = np.squeeze(image, axis=0)

    return image, prediction

def display_all(i, count):
    plt.figure(figsize=(15, 5))

    end_index = min(i + count, len(images_files))

    for index in range(i, end_index):
        filename = images_files[index]
        image, prediction = get_prediction(filename=filename)

        # Plot the image
        plt.subplot(2, count, (index - i) + 1)
        plt.imshow(image)
        plt.title(f"Image {index + 1}")
        plt.axis('off')

        # Plot the corresponding label
        plt.subplot(2, count, (index - i) + 1 + count)
        plt.imshow(np.squeeze(prediction), cmap='gray')
        plt.title(f"Prediction {index + 1}")
        plt.axis('off')
    
    plt.show()

def display_one(i):
    filename = images_files[i]
    
    image, prediction = get_prediction(filename=filename)

    plt.figure(figsize=(10, 5))

    # Plot input image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    # Plot predicted edges
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(prediction), cmap='gray')
    plt.title('Predicted Edges')
    plt.axis('off')

    plt.show()

# display_all(10, 20)
display_one(2)