import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype('float32')
print(x_train[1:,:].shape)
x_test = x_test[..., tf.newaxis].astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/mnist_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)

# Test model on random input data.
input_shape = input_details[0]['shape']
print(f"Input shape: {input_shape}")
output_shape = output_details[0]['shape']
print(f"Output shape: {output_shape}")
input_data = x_test[1:2,:]
print(input_data.shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(y_test[1])
print(tf.argmax(output_data, axis=1))