import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import cv2
import numpy as np
import time
import tensorflow as tf
import info

model_num = 7
path = info.get_path(model_num)

interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]  # Model's expected input size

print("input: ", input_details[0]['dtype'])
print("output", output_details[0]['dtype'])

cap = cv2.VideoCapture(0)
count = 0

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV default) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # rgb_frame = cv2.flip(rgb_frame, 1)

    resized_frame = cv2.resize(rgb_frame, (width, height))
    # input_data = resized_frame.astype('uint8')
    input_data = np.expand_dims(resized_frame, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.squeeze(output_data) # Remove batch dimension
    output_data = (output_data - output_data.min()) / (output_data.max() - output_data.min())
    # output_data = output_data.astype('uint8')

    # print(output_data.shape)
    output_data = cv2.resize(output_data, (512, 512))

    end = time.time()
    fps = math.ceil(1 / (end - start))
    count += 1
    print(f"{fps} fps")

    if count == 1:
        average_fps = fps
    else:
        average_fps = math.ceil((average_fps * count + fps) / (count + 1))

    cv2.imshow("Edge Detection", output_data)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
