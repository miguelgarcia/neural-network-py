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

from neuralnetwork.network import NeuralNetwork, serialize_neural_network
from neuralnetwork.training import Trainer
from mnist import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digits_recognizer")

# Create neural network
neural_network = NeuralNetwork([784, 20, 10])

# # Train the network
logger.info("Begining training")
logger.info("Loading training dateset")
training_data = load_dataset('./datasets/mnist/mnist_train.csv')
logger.info("Training dateset loaded")

logger.info("Training network")
trainer = Trainer()
trainer.train(
    network=neural_network,
    training_data=training_data,
    learning_step=1,
    batch_iterations=20,
    batch_size=50,
    min_improvement_per_batch=0.000001,
    max_batches_without_improvement=100,
    cost_estimator_batch_size=5000
)

logger.info("Network has been trained")
logger.info("Serializing network setup to digits_recognizer_setup.txt")
with open("digits_recognizer_setup.txt", "w") as f:
    serialize_neural_network(neural_network, f)
