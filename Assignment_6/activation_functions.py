import numpy as np


# activation functions and their derivative
def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return np.where(x>=0, 1, 0)


def linear(x):
    return x


def linear_derivative(x):
    return 1


activation = {'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu, 'linear':linear}
activation_derivative = {'sigmoid': sigmoid_derivative, 'tanh': tanh_derivative, 'relu': relu_derivative, 'linear':linear}


###########################################
#       Test part of activations          #
#       functions and their derivative    #
###########################################
if __name__ == '__main__':
    print('relu')
    print(activation['relu'](np.array([4, 0, -4])))
    print(activation_derivative['relu'](np.array([4, 0, -4])))

    print('sigmoid')
    print(activation['sigmoid'](np.array([4, 0, -4])))
    print(activation_derivative['sigmoid'](np.array([4, 0, -4])))

    print('tanh')
    print(activation['tanh'](np.array([4, 0, -4])))
    print(activation_derivative['tanh'](np.array([4, 0, -4])))

