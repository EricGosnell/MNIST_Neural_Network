# Numpy is used for the majority of matrix operations, matplotlib is for visuals
import numpy as np
# import matplotlib.pyplot as plt

# Using pandas and sklearn for data fetching and initial preprocessing
# import pandas as pd
from sklearn import datasets

# Fetching the data from sklearn.datasets:
X, y = datasets.fetch_openml('mnist_784',
                             version=1,
                             as_frame=True,
                             return_X_y=True,
                             parser='auto')
X = np.array(X).reshape(70000, 784, 1) / 255.0
y = np.array(y, dtype='int')


# Encodes categorical variables as one-hot vector,
#   e.g. 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def one_hot(v):
    one_hot_vector = np.zeros((v.size, v.max() + 1))
    one_hot_vector[np.arange(v.size), v] = 1
    return one_hot_vector


y = one_hot(y)
X_train, X_test = X[5000:, :, :], X[:5000, :, :]
y_train, y_test = y[5000:, :], y[:5000, :]


# Layer:
#   Used as parent class for all layers in network architecture.
#   Defines input and output attributes, as well as forward and backward propagation functions
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Forward propagation function to be specified for each layer
    def forward_prop(self, input):
        pass

    # Backward propagation function to be specified for each layer
    def backward_prop(self, output_gradient, learning_rate):
        pass


# Dense Layer:
#   Used as the dense layer object in building neural network models
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Initializes weight matrix and bias vector with elements from a standard normal distribution
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)  # Dimensions j x i
        self.biases = np.random.randn(output_size, 1)  # Dimensions j x 1

    # Forward Propagation:
    # Computes forward propagation of the dense layer as described by the matrix equation Y = WX + b
    def forward_prop(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    # Backward Propagation:
    #   Inputs:
    #   output_gradient - The gradient of the cost function with respect to the output of this layer
    #   learning_rate - The learning rate as specified by the training function
    #
    #   Updates the weight matrix and bias vector and passes the gradient of the input back to the previous layer
    def backward_prop(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


# Activation Layer:
#   Used as the activation function for each layer
class Activation(Layer):
    # Initializes activation function and its derivative which are specified by the specific activation function classes
    def __init__(self, activation, d_activation):
        super().__init__()
        self.activation = activation
        self.d_activation = d_activation

    # Forward Propagation:
    # Computes the activation function for each node passed by the previous layer
    def forward_prop(self, input):
        self.input = input
        return self.activation(input)

    # Backward Propagation:
    #   Inputs:
    #   output_gradient - The gradient of the cost function with respect to the output of this layer
    #   learning_rate - The learning rate as specified by the training function
    #
    # Updates the weight matrix and bias vector and passes the gradient of the input back to the previous layer
    def backward_prop(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.d_activation(self.input))


# ReLU:
#   'Rectified Linear Unit' activation function, which computes the piecewise function f(x) = max{0,x}
class ReLU(Activation):
    def __init__(self):
        # np.maximum() performs element-wise max operation, as opposed to np.max()
        relu = lambda x: np.maximum(0, x)
        # Derivative is 1 for positive x, 0 for negative x:
        d_relu = lambda x: (x > 0)
        # Initializing Activation layer with ReLU:
        super().__init__(relu, d_relu)


# Softmax:
#   Computes the softmax() function, scaling all elements of X to a value between 0 and 1, each of which can be
#   interpreted as the probability of a given classification
class Softmax(Layer):
    def forward_prop(self, input):
        # Shifting all input values to avoid overflow errors
        input = input - np.max(input)
        self.output = np.exp(input) / np.sum(np.exp(input))
        return self.output

    def backward_prop(self, output_gradient, learning_rate):
        # Recall that self.output is just the function applied to the input,
        # the '-' and '*' operators perform the element-wise operation on the axis it shares with the given vector
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


# mse: Computes the mean-squared error between prediction and observation
def mse(y_pred, y_true):
    return np.mean(np.power(y_true - y_pred, 2))


# d_mse: Computes the derivative of the mean-squared error for a prediction
def d_mse(y_pred, y_true):
    return 2 * (y_pred.T - y_true).T / np.size(y_true)


class Network:
    def __init__(self, layers):
        self.layers = layers

    # Predict:
    #   Given an input, makes a prediction by propagating through each layer
    def predict(self, input):
        output = input

        # Loop through layers, repeatedly updating output
        for layer in self.layers:
            output = layer.forward_prop(output)

        return output

    def train(self, X, y, cost, d_cost, validation_split=.2, epochs=30, batch_size=32, learning_rate=.01, verbose=True):
        history = {'Accuracy': [], 'Validation Accuracy': []}
        size = X.shape[0]

        # Splitting data into training set and validation set for CV
        split = int(validation_split * size)
        X_train, X_val = X[split:], X[:split]
        y_train, y_val = y[split:], y[:split]

        for epoch in range(epochs):
            count = 0
            val_count = 0
            # z = np.random.permutation(batch_size)

            for x, y in zip(X_train, y_train):

                # Forward Pass:
                output = self.predict(x)

                # Backward Pass:
                grad = d_cost(output, y)
                for layer in reversed(self.layers):
                    grad = layer.backward_prop(grad, learning_rate)

                if np.argmax(output) == np.argmax(y):
                    count += 1

            # Computing training accuracy
            accuracy = count / (size - split)

            # Computing validation accuracy
            for x, y in zip(X_val, y_val):
                output = self.predict(x)
                if np.argmax(output) == np.argmax(y):
                    val_count += 1

            val_accuracy = val_count / split

            if verbose:
                print('Epoch: %d/%d' % (epoch + 1, epochs))
                print(f'Training Accuracy: {accuracy:.2%} - Validation Accuracy: {val_accuracy:.2%}')

            # Stores accuracy at each epoch
            history['Accuracy'].append(accuracy)
            history['Validation Accuracy'].append(val_accuracy)

        return history


# Initializing network object
network = Network([
    Dense(784, 40),
    ReLU(),
    Dense(40, 10),
    Softmax()
])

history = network.train(X_train, y_train, mse, d_mse, epochs=50, batch_size=32, learning_rate=.03)

# Computing test accuracy
right = 0
preds = []

for x, y in zip(X_test, y_test):
    output = np.argmax(network.predict(x))
    preds.append(output)
    if output == np.argmax(y):
        right += 1

print('Test Accuracy:', right / X_test.shape[0])

# history = pd.DataFrame(history)
# history.plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.title('30 Epochs of Training')
# plt.show()
