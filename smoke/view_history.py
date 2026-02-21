import pickle
import matplotlib.pyplot as plt
import information
import os
import glob
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from tensorflow.keras.models import load_model
from PIL import Image

history_files = glob.glob("saved_histories/dehazer_history_trial*.pkl")

def preprocess_image(image_path, image_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(image_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def calculate_psnr_for_trial(trial_num, model):
    hazy_dir = f"SMOKE/train/hazy"
    ground_truth_dir = "SMOKE/train/clean"  
    
    hazy_files = sorted(glob.glob(os.path.join(hazy_dir, "*.png")))
    ground_truth_files = sorted(glob.glob(os.path.join(ground_truth_dir, "*.png")))
    
    hazy_path = hazy_files[0]
    gt_path = ground_truth_files[0]
    
    hazy_image = preprocess_image(hazy_path)
    ground_truth = Image.open(gt_path).convert('RGB')
    ground_truth = ground_truth.resize((256, 256))
    ground_truth = np.array(ground_truth) / 255.0
    
    dehazed_image = model.predict(hazy_image)
    dehazed_image = np.squeeze(dehazed_image, axis=0)
    
    psnr_value = psnr(ground_truth, dehazed_image, data_range=1.0)
    
    return psnr_value  

psnr_results = {}
for file_path in history_files:
    trial_part = file_path.split('trial')[-1].split('.')[0]
    trial_num = int(trial_part.replace('_', ''))
    
    # Load the model for the current trial
    model_path = f'saved_models/dehazer_trial_{trial_num}.keras'
    model = load_model(model_path)
    
    psnr_value = calculate_psnr_for_trial(trial_num, model)
    psnr_results[trial_num] = psnr_value
    print(f"Trial {trial_num} PSNR: {psnr_value:.2f} dB")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

for file_path in history_files:
    trial_part = file_path.split('trial')[-1].split('.')[0]
    trial_num = int(trial_part.replace('_', ''))
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    ax1.plot(history['mae'], label=f'MAE (Trial {trial_num})')

ax2.bar(psnr_results.keys(), psnr_results.values(), label='PSNR')

ax1.set_title("Mean Absolute Error Across All Trials")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MAE")
ax1.legend()

ax2.set_title("PSNR Values Across All Trials")
ax2.set_xlabel("Trial")
ax2.set_ylabel("PSNR (dB)")
ax2.legend()

plt.tight_layout()
plt.show()

print("\nMaximum validation accuracies:")
for file_path in history_files:
    trial_part = file_path.split('trial')[-1].split('.')[0]
    trial_num = int(trial_part.replace('_', ''))
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    max_val_acc = max(history['val_mae'])
    print(f"Trial {trial_num} Maximum Validation MAE: {max_val_acc:.4f}")