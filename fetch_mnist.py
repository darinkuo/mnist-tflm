import tensorflow as tf

class MNISTDataset:
    """
    MNIST Dataset retrieval and loading
    """
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.samples = self.x_train.shape[0]
        self.dimensions = (self.x_train.shape[1], self.x_train.shape[2])
