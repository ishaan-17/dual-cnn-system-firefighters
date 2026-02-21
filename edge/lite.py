import tensorflow as tf
import information
import lite_info
import numpy as np

path = information.MODEL_SAVE_PATH
model = tf.keras.models.load_model(path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():
    for _ in range(100):
        data = np.random.rand(1, 256, 256, 3).astype('float32')
        yield [data]

converter.representative_dataset = representative_dataset_gen

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open(lite_info.MODEL_SAVE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Model Version {lite_info.VERSION}")