import os

import tensorflow as tf

from utils import rep_data_gen, write_tf_lite_micro_model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create an instance of the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=2,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])

model.summary()

# Save the model
print("Saving modeling in path mnist-model...")

# Create the model folder if it doesn't exist
if not os.path.exists('model'):
    os.mkdir('model')

model.save('model/mnist_model.h5')

print("Model saved...")
print("Converting to TFLite Flatbuffer...")
# Converter flags and optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = lambda: rep_data_gen(x_test)
tflite_quant_model = converter.convert()

# Save the flatbuffer
with open('mnist_quant.tflite', "wb") as f:
    f.write(tflite_quant_model)

# Generate the C header and source file
write_tf_lite_micro_model(tflite_quant_model, data_variable_name="mnist_model_data")
