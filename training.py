import numpy as np


class TrainingDataSample:
    def __init__(self, input_values, output_values):
        self.input = input_values
        self.output = output_values


class Trainer:
    """
    Train a neural network using stochastic gradient descent
    """
    def make_batch(self, samples, size):
        batch = []
        for _ in range(0, size):
            batch.append(samples[np.random.randint(0, len(samples))])
        return batch

    def calculate_cost(self, network, batch):
        cost = 0
        for sample in batch:
            out = network.feedforward(sample.input)
            tmp = out - sample.output
            cost += (tmp*tmp).sum()
        return cost

    def train(self, network, training_data, learning_step, iterations, batch_size, min_improvement=0, max_batches_without_improvement=3, cost_estimator_batch_size=None, show_progress=False):
        # TODO:Starting from multiple configurations
        work = max_batches_without_improvement # For at most for 3 batches without improvement
        it = 0
        if cost_estimator_batch_size is None:
            cost_estimator_batch = training_data
        else:
            cost_estimator_batch = self.make_batch(training_data, cost_estimator_batch_size)
        min_cost = None
        while work > 0:
            # Make batch
            it += 1
            batch = self.make_batch(training_data, batch_size)
            cost = self.calculate_cost(network, cost_estimator_batch)
            if min_cost is None:
                min_cost = cost
            if show_progress: # and it % 100 == 0
                print("Working on batch %d %f" % (it, min_cost))
            for i in range(0, iterations):
                gradients = []
                # Calculate gradients for each sample
                for sample in batch:
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
            new_cost = self.calculate_cost(network, cost_estimator_batch)
            if new_cost < min_cost * (1-min_improvement):
                work = max_batches_without_improvement
                min_cost = new_cost
            else:
                work -=1
