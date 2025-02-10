import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size, 4)  # 4 outputs for movement directions
        self.bias = np.random.randn(4)
        self.learning_rate = learning_rate

    def activation_function(self, x):
        """ Step function for classification """
        return np.where(x > 0, 1, 0)

    def predict(self, inputs):
        """ Forward pass: Compute output from inputs """
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, inputs, target):
        """ Train the perceptron using supervised learning """
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights += self.learning_rate * np.outer(inputs, error)
        self.bias += self.learning_rate * error
