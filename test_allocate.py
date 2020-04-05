import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="./mnist_model.tflite")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details[0]['index'])
interpreter.resize_tensor_input(input_details[0]['index'], (5, 784))
interpreter.resize_tensor_input(output_details[0]['index'], (5, 10))
#interpreter.resize_tensor_input(input_details[0]["index"], [batch_size, 513, 513, 3])
interpreter.allocate_tensors()