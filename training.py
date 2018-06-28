import numpy as np


class TrainingDataSample:
    def __init__(self, input_values, output_values):
        self.input = input_values
        self.output = output_values


class Trainer:
    """
    Train a neural network using stochastic gradient descent
    """

    def train(self, network, training_data, learning_step, iterations, batch_size):
        # TODO:Starting from multiple configurations
        for i in range(0, iterations):
            gradients = []
            # Calculate gradients for each sample
            for _ in range(0, batch_size):
                sample = training_data[np.random.randint(
                    0, len(training_data))]
                g = network.backprop(sample.input, sample.output)
                if gradients == []:
                    gradients = g
                else:
                    for g_i in range(0, len(g)):
                        gradients[g_i][0] += g[g_i][0]
                        gradients[g_i][1] += g[g_i][1]
            # Average gradients and update network
            layer_n = network.layers_count()-1
            for g in gradients:
                network.dec_weights(layer_n, learning_step *
                                    (g[0] / float(batch_size)))
                network.dec_biases(layer_n, learning_step *
                                   (g[1] / float(batch_size)))
                layer_n -= 1
