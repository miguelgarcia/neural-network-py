import textwrap

import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# def reLu(z):
#     return np.maximum(x, 0)

# def reLu_prime(z):
#     return (x > 0).astype(float)


def activation(x):
    return sigmoid(x)


def activationDerivative(x):
    return sigmoid_prime(x)


class NeuralLayer:
    """
    Layer of a neural network, contains a configurable number of neurons.
    Each neuron in this layer has associated a bias and many weights, one for
    each neuron in the previous layer in the network.
    """

    def __init__(self, neuron_count, prev_layer_neuron_count):
        self.weights = np.random.random(
            (neuron_count, prev_layer_neuron_count))
        self.biases = np.random.random(neuron_count)
        self.output = np.zeros(neuron_count)
        self.neuron_count = neuron_count

    def feed(self, input_values):
        """
        Calculates this layer output based on the input values (the output 
        values of the previous layer)
        """
        self.output = activation(self.weights.dot(input_values) + self.biases)
        return self.output

    def __str__(self):
        rep = "<NeuralLayer>\n"
        rep += "  <Weights>\n"
        rep += textwrap.indent(str(self.weights), "    ") + "\n"
        rep += "  </Weights>\n"
        rep += "  <Biases>\n"
        rep += textwrap.indent(str(self.biases), "    ") + "\n"
        rep += "  </Biases>\n"
        rep += "  <Output>\n"
        rep += textwrap.indent(str(self.output), "    ") + "\n"
        rep += "  </Output>\n"
        rep += "</NeuralLayer>"
        return rep


class NeuralInputLayer(NeuralLayer):
    """
    The input layer has no weights and biases
    """

    def __init__(self, neuron_count):
        self.neuron_count = neuron_count
        self.output = np.zeros(neuron_count)

    def feed(self, input_values):
        self.output = input_values
        return self.output

    def __str__(self):
        rep = "<NeuralInputLayer>\n"
        rep += "  <Output>\n"
        rep += textwrap.indent(str(self.output), "    ") + "\n"
        rep += "  </Output>\n"
        rep += "</NeuralInputLayer>"
        return rep


class NeuralNetwork:
    """
    Layered neural network with configurable shape (layers count and neurons 
    per layer)
    """

    def __init__(self, shape):
        """ shape: list of neurons count per layer """
        self.layers = []
        self.layers.append(NeuralInputLayer(shape[0]))
        for i in range(1, len(shape)):
            self.layers.append(NeuralLayer(shape[i], shape[i-1]))

    def layers_count(self):
        return len(self.layers)

    def __str__(self):
        rep = "<NeuralNetwork>\n"
        for layer in self.layers:
            rep += textwrap.indent(str(layer), "  ") + "\n"
        rep += "</NeuralNetwork>"
        return rep

    def feedforward(self, input_values):
        """ Calculate final layer result

        L_i = activation(weights[i-1] * L_(i-1) + biases[i-1])
        """
        output = input_values
        for layer in self.layers:
            output = layer.feed(output)
        return output

    def dec_weights(self, layer_n, weights_decrement):
        self.layers[layer_n].weights -= weights_decrement

    def dec_biases(self, layer_n, biases_decrement):
        self.layers[layer_n].biases -= biases_decrement

    def backprop(self, input_values, expected_result):
        """
        Calculates the gradient of the cost function.
        Returns a list of tuples [(wL,bL),...,(w1,b1)]
          wi is a matrix with the gradient for each weight in layer i.
          bi is an array with the gradient for each bias in layer i.

        Note that gradient info for the last layer appears first in the returned list.
        """
        # See http://neuralnetworksanddeeplearning.com/chap2.html
        # feed forward
        actual_result = self.feedforward(input_values)
        gradients = []

        next_layer_delta = None
        for i in range(len(self.layers)-1, 0, -1):
            # for the output and each hidden layer in reverse order
            # prevLayer -> layer -> nextLayer
            layer = self.layers[i]
            prev_layer = self.layers[i-1]
            z = layer.weights.dot(prev_layer.output) + layer.biases
            if i == len(self.layers) - 1:
                # Last layer
                delta = (actual_result - expected_result) * \
                    activationDerivative(z)
            else:
                # Hidden layer
                next_layer = self.layers[i+1]
                delta = next_layer.weights.T.dot(
                    next_layer_delta) * activationDerivative(z)
            o_t = prev_layer.output.reshape((-1, 1)).T
            delta_t = delta.reshape((-1, 1))
            w = delta_t.dot(o_t)
            b = delta
            gradients.append([w, b])
            next_layer_delta = delta
        return gradients
