import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

save_path = os.path.dirname(os.path.realpath(__file__)) + "/models/number_guess_model"
model = tf.keras.models.load_model(save_path)

predictions = model.predict(np.array(test_images))
print(np.array(test_images)[1])
print(np.argmax(predictions[1]))

# Show the image in the set that was predicted upon
# Uncomment if xming and mathplotlib is installed and 
# export DISPLAY=localhost:0.0 is included in .bashrc
plt.imshow(test_images[1], cmap = plt.cm.binary)
plt.show()
