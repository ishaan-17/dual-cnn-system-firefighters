import tensorflow as tf
import information

path = information.MODEL_SAVE_PATH
model = tf.keras.models.load_model(path)
model.summary()