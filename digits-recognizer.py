#!/usr/bin/env python3
# Handwritten digits recognizer
#
# Input: 28x28 pixels
# Output: 10x1 array. output[i] = probability of the input being the digit i
#
# Trained and tested using the MNIST dataset
#
import logging
import os
import sys

import numpy as np

from neuralnetwork import NeuralNetwork, serialize_neural_network, load_from_file
from training import Trainer, TrainingDataSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digits_recognizer")

# -- MNIST dataset parsing and result conversion functions --
results_table = np.identity(10)


def load_dataset(path):
    samples = []
    with open(path, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split(',')))
            inp = np.array(data[1:]) / 255
            r = results_table[int(data[0])]
            samples.append(TrainingDataSample(inp, r))
    return samples


def result_to_digit(output):
    return np.argmax(output)
# -----------------------------------------------------------


# # Create neural network
# neural_network = NeuralNetwork([784, 20, 10])

# # Train the network
# logger.info("Begining training")
# logger.info("Loading training dateset")
# training_data = load_dataset('./datasets/mnist/mnist_train.csv')
# logger.info("Training dateset loaded")

# logger.info("Training network")
# trainer = Trainer()
# trainer.train(
#     network=neural_network,
#     training_data=training_data,
#     learning_step=1,
#     batch_iterations=20,
#     batch_size=50,
#     min_improvement_per_batch=0.000001,
#     max_batches_without_improvement=100,
#     cost_estimator_batch_size=5000
# )
# del training_data

# logger.info("Network has been trained")
# logger.info("Serializing network setup to digits_recognizer_setup.txt")
# with open("digits_recognizer_setup.txt", "w") as f:
#     serialize_neural_network(neural_network, f)
with open("digits_recognizer_setup.txt", "r") as f:
    neural_network = load_from_file(f)
logger.info("Testing")
logger.info("Loading testing dateset")
testing_data = load_dataset('./datasets/mnist/mnist_test.csv')
logger.info("Testing dateset loaded")

logger.info("Testing network")
good = 0
bad = 0
case_index = 0
for sample in testing_data:
    result = neural_network.feedforward(sample.input)
    expected_result = result_to_digit(sample.output)
    actual_result = result_to_digit(result)
    print("Case %d: %d %d %d" % (case_index, actual_result,
                                 expected_result, actual_result == expected_result))
    if actual_result == expected_result:
        good += 1
    else:
        bad += 1
    case_index += 1

print("Good: %d Bad: %d Success Rate: %f" %
      (good, bad, float(good) / float(good+bad)))
