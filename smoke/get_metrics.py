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
dataset_path = 'SMOKE'
train_image_path = dataset_path + '/train/hazy/'
train_label_path = dataset_path + '/train/clean/'

test_image_path = dataset_path +  '/test/hazy/'
test_label_path = dataset_path +  '/test/clean/'

train_images = sorted([f for f in os.listdir(train_image_path)])
train_labels = sorted([f for f in os.listdir(train_label_path)])

image_size = (256, 256)
label_size = (1280, 720)
MODEL_SAVE_PATH = information.MODEL_SAVE_PATH

def load_image(image_path, label_path, size=(256, 256)):
    """
    Load and preprocess an RGB image and its corresponding ground truth.
    Args:
        image_path (str): Path to the input RGB image.
        label_path (str): Path to the ground truth edge map.
        size (tuple): Target size for resizing (default is 256x256).
    Returns:
        image (numpy.ndarray): Preprocessed input RGB image.
        label (numpy.ndarray): Preprocessed ground truth edge map.
    """
    # Load and resize the input RGB image
    image = cv.imread(image_path)
    image = cv.resize(image, size).astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Load and resize the ground truth edge map
    label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
    label = cv.resize(label, label_size).astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    return image, label

def calculate_ssim(predicted, ground_truth):
    """
    Calculate SSIM between predicted and ground truth images.
    Args:
        predicted (numpy.ndarray): Predicted edge map (float32, 256x256).
        ground_truth (numpy.ndarray): Ground truth edge map (float32, 256x256).
    Returns:
        ssim_value (float): SSIM score.
    """
    ssim_value, _ = ssim(ground_truth, predicted, full=True, data_range=1.0)
    return ssim_value

def calculate_iou(predicted, ground_truth, threshold=0.5):
    """
    Calculate IoU between predicted and ground truth images.
    Args:
        predicted (numpy.ndarray): Predicted edge map (float32, 256x256).
        ground_truth (numpy.ndarray): Ground truth edge map (float32, 256x256).
        threshold (float): Threshold value for binarization.
    Returns:
        iou (float): IoU score.
    """
    # Binarize the predicted and ground truth edge maps
    predicted_binary = (predicted > threshold).astype(np.uint8)
    ground_truth_binary = (ground_truth > threshold).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(predicted_binary, ground_truth_binary)
    union = np.logical_or(predicted_binary, ground_truth_binary)
    
    # Sum all True pixels to get the count
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)
    
    # Avoid division by zero
    if union_count == 0:
        return 0.0
    
    # Calculate IoU
    iou = intersection_count / union_count
    return iou

# Example usage
if __name__ == "__main__":
    # Load the model
    model = load_model(MODEL_SAVE_PATH)
    
    # Initialize accumulators for SSIM and IoU
    total_ssim = 0.0
    total_iou = 0.0
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

        '''plt.figure(figsize=(10, 5))

        # Plot input image
        plt.subplot(1, 2, 1)
        plt.imshow(predicted, cmap='gray')
        plt.axis('off')

        # Plot predicted edges
        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth, cmap='gray')
        plt.axis('off')

        plt.show()

        exit()'''
        
        # Calculate SSIM and IoU
        ssim_value = calculate_ssim(predicted, ground_truth)
        iou_value = calculate_iou(predicted, ground_truth)
        
        # Accumulate the results
        total_ssim += ssim_value
        total_iou += iou_value
        
        # Print progress
        print(f"Processed image {i + 1}/{num_images}: SSIM = {ssim_value:.4f}, IoU = {iou_value:.4f}")
    
    # Calculate average SSIM and IoU
    avg_ssim = total_ssim / num_images
    avg_iou = total_iou / num_images
    
    print(f"\nAverage SSIM over {num_images} images: {avg_ssim:.4f}")
    print(f"Average IoU over {num_images} images: {avg_iou:.4f}")