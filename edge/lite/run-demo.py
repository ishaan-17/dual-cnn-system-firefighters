import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import cv2
import numpy as np
import time
import tensorflow as tf
import info

interpreter = tf.lite.Interpreter(model_path=info.get_path(0))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]  # Model's expected input size

SIZE = 256
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE)

count = 0

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV default) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.flip(rgb_frame, 1)

    # Preprocess input
    resized_frame = cv2.resize(rgb_frame, (width, height))  # Resize to model input size
    input_data = resized_frame.astype('float32') / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # Inference / Prediction
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process Output
    output_data = np.squeeze(output_data)  # Remove batch dimension
    output_data = (output_data - output_data.min()) / (output_data.max() - output_data.min()) * 255
    output_data = output_data.astype('uint8')
    output_data = cv2.resize(output_data, (512, 512))

    end = time.time()
    fps = math.ceil(1 / (end - start))
    count += 1
    print(f"{fps} fps")

    if count == 1:
        average_fps = fps
    else:
        average_fps = math.ceil((average_fps * count + fps) / (count + 1))

    cv2.putText(frame, f"fps: {str(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
    cv2.putText(frame, f"average fps: {str(average_fps)}", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    cv2.imshow("Edge Detection", output_data)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
