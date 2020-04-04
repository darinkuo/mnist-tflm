import tensorflow as tf
import matplotlib.pyplot as plt
import os

mnist = tf.keras.datasets.mnist # 28x28 image of hand-written digits 0-9

# Seperate the training and testing datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# retrieve the input dimensions and print the number of training and test data
print(train_images.shape[0], test_images.shape[0])

data_dim = (train_images[1], train_images[2])

# Normalize the training and test data
train_images = tf.keras.utils.normalize(train_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)

# Use matplotlib to visualize testing data
# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()
# plt.imshow(x_test[0], cmap = plt.cm.binary)
# plt.show()

# Print the array representing the data
# print(x_train[0])
# print(x_test[0])

# Build the Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
				loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])

# Feed the model training data
model.fit(train_images, train_labels, epochs = 10)

# Evaluate accuracy
val_loss, val_acc = model.evaluate(test_images, test_labels)
print(val_loss, val_acc)

# Export the model to a SavedModel format for TFLite conversion
save_path = os.path.dirname(os.path.realpath(__file__)) + "/models/number_guess_model"
model.save(save_path, save_format='tf')
