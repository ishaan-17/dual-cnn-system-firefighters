import pickle
import matplotlib.pyplot as plt

TRIAL_NUMBER = 3
path = f"saved_histories/training_history_trial_{TRIAL_NUMBER}.pkl"

# Load history from the specified trial file
with open(path, 'rb') as f:
    history = pickle.load(f)

# Plot the training and validation accuracy
plt.figure(figsize=(10,6))
plt.plot(history['loss'], label='Training Accuracy')
plt.plot(history['val_loss'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy: Trial " + str(TRIAL_NUMBER))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

print(max(history['val_accuracy']))