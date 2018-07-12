import numpy as np
import logging

logger = logging.getLogger("training")


class TrainingDataSample:
    def __init__(self, input_values, output_values):
        self.input = input_values
        self.output = output_values


class Trainer:
    """
    Train a neural network using stochastic gradient descent
    """

    def make_batch(self, samples, size):
        """ Takes a sample of size elements from samples """
        batch = []
        for _ in range(0, size):
            batch.append(samples[np.random.randint(0, len(samples))])
        return batch

    def calculate_cost(self, network, batch):
        """ Calculate the cost for a list of samples 
            Cost = sum for each sample (expected_result - actual_result)^2
        """
        cost = 0
        for sample in batch:
            out = network.feedforward(sample.input)
            tmp = out - sample.output
            cost += (tmp*tmp).sum()
        return cost

    def train(self, network, training_data, learning_step, batch_iterations, batch_size, min_improvement_per_batch=0, max_batches_without_improvement=3, cost_estimator_batch_size=None):
        """ Train the network using stocastic gradient descent method

        Algorithm:
        0- Build a batch used for cost estimation, if cost_estimator_batch_size
            is None, use all the samples in training data.
        1- Estimate current cost for cost estimation batch
        2- Pick a batch of batch_size samples from training_data
        3- Repeat batch_iterations:
            3.1- Calculate cost function gradient for each sample
            3.2- Average the gradients
            3.3- Scale gradients by learning_step
            3.4- Update networks weights and biases substracting the gradients
        4- Estimate updated cost for cost estimation batch
        5- The picked batch produced an improvement if 
            new_cost < min_cost * (1-min_improvement_per_batch)
        6- Repeat until max_batches_without_improvement consecutive batches are
            processed without making improvement (as defined in 5).
        """
        # For at most for max_batches_without_improvement batches without improvement
        pending_batches = max_batches_without_improvement
        # Build batch used to estimate cost
        if cost_estimator_batch_size is None:
            cost_estimator_batch = training_data
        else:
            cost_estimator_batch = self.make_batch(
                training_data, cost_estimator_batch_size)
        it = 0
        min_cost = None
        while pending_batches > 0:
            # Work until max_batches_without_improvement iterations without
            # an improvement of (min_improvement_per_batch * 100)% are reached

            # Current cost
            cost = self.calculate_cost(network, cost_estimator_batch)
            if min_cost is None:
                min_cost = cost

            # Make a batch and use it to make batch_iterations iterations of
            # stocastic gradient descent
            batch = self.make_batch(training_data, batch_size)
            logger.info("Working on batch %d %f" % (it, min_cost))
            for _ in range(0, batch_iterations):
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
                # Multiplying average gradients by learning_step
                layer_n = network.layers_count()-1
                for g in gradients:
                    network.dec_weights(layer_n, learning_step *
                                        (g[0] / float(batch_size)))
                    network.dec_biases(layer_n, learning_step *
                                       (g[1] / float(batch_size)))
                    layer_n -= 1
            # Calculate new cost
            new_cost = self.calculate_cost(network, cost_estimator_batch)
            # Check if there was an improvement of at least
            # (min_improvement_per_batch * 100)%
            if new_cost < min_cost * (1-min_improvement_per_batch):
                pending_batches = max_batches_without_improvement
                min_cost = new_cost
            else:
                pending_batches -= 1
            it += 1
