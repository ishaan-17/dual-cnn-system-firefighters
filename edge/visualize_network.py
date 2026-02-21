from tf_explain.core import GradCAM
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import information

# Load the model
path = information.MODEL_SAVE_PATH
model = load_model(path)

# Load and preprocess the image
image_path = 'prediction_images/5.jpg'
img = Image.open(image_path)
img = img.resize((256, 256))  # Resize to match your model's input size
x_test = np.array(img) / 255.0  # Normalize

# Create explainer
explainer = GradCAM()

# Specify the last convolutional layer for GradCAM
# You'll need to replace 'conv2d_X' with your actual last conv layer name
last_conv_layer = model.get_layer('conv2d_2')  # Adjust layer name as needed

# Generate the explanation
grid = explainer.explain((np.expand_dims(x_test, axis=0), None), 
                        model, 
                        class_index=0,
                        layer_name=last_conv_layer.name)

plt.imshow(grid, cmap='jet')
plt.show()