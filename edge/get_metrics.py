# imports
import os
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import information

# Remove TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Dataset paths
dataset_path = 'BIPEDv2/BIPED/edges'
train_image_path = dataset_path + '/imgs/train/rgbr/real/'
train_label_path = dataset_path + '/edge_maps/train/rgbr/real/'
test_image_path = dataset_path +  '/imgs/test/rgbr/'
test_label_path = dataset_path +  '/edge_maps/test/rgbr/'

train_images = sorted([f for f in os.listdir(train_image_path)])
train_labels = sorted([f for f in os.listdir(train_label_path)])

image_size = (256, 256)
label_size = (1280, 720)
MODEL_SAVE_PATH = information.MODEL_SAVE_PATH

def load_image(image_path, label_path, size=(256, 256)):
    # Load and resize the input RGB image
    image = cv.imread(image_path)
    image = cv.resize(image, size).astype(np.float32) / 255.0 
    
    # Load and resize the ground truth edge map
    label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
    label = cv.resize(label, label_size).astype(np.float32) / 255.0 
    
    return image, label

def calculate_ssim(predicted, ground_truth):
    ssim_value, _ = ssim(ground_truth, predicted, full=True, data_range=1.0)
    return ssim_value

# Example usage
if __name__ == "__main__":
    # Load the model
    model = load_model(MODEL_SAVE_PATH)
    
    # Initialize accumulators for SSIM
    total_ssim = 0.0
    num_images = len(train_images)
    
    # Loop through the train dataset
    for i in range(num_images):
        # Load the input RGB image and ground truth
        image_path = os.path.join(train_image_path, train_images[i])
        label_path = os.path.join(train_label_path, train_labels[i])
        image, ground_truth = load_image(image_path, label_path)
        
        # Predict the edge map using the model
        image_input = np.expand_dims(image, axis=0)  # Add batch dimension
       
        predicted = model.predict(image_input)
        predicted = np.squeeze(predicted)  # Remove batch dimension
        # predicted = predicted.resize((256, 256))
        predicted = cv.resize(predicted, (label_size), interpolation=cv.INTER_LINEAR)
        
        # Calculate SSIM
        ssim_value = calculate_ssim(predicted, ground_truth)
        
        # Accumulate the results
        total_ssim += ssim_value
        
        # Print progress
        print(f"SSIM = {ssim_value:.4f}")
    
    # Calculate average SSIM
    avg_ssim = total_ssim / num_images
    
    print(f"\nSSIM: {avg_ssim:.4f}")