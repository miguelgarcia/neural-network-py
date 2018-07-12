#!/usr/bin/env python3
# Mini example, classify points in two teams:
# "team 1" if x > y, "team 0" otherwise
import logging

import numpy as np

from neuralnetwork.network import NeuralNetwork, serialize_neural_network
from neuralnetwork.training import Trainer, TrainingDataSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_example")

# Use a network with two layers:
# Input has 2 neurons
# Output has 1 neuron
neural_network = NeuralNetwork([2, 1])

# Calculate the correct result for a point


def team(p):
    if p[0] > p[1]:
        return 1
    return 0


logger.info("Generating training data")
training_data = []
for _ in range(0, 500):
    p = np.random.random(2)
    training_data.append(TrainingDataSample(p, np.array([team(p)])))


logger.info("Training")
trainer = Trainer()
trainer.train(
    network=neural_network,
    training_data=training_data,
    learning_step=1,
    batch_iterations=10,
    batch_size=10,
    min_improvement_per_batch=0.00001,
    max_batches_without_improvement=15,
    cost_estimator_batch_size=None
)
logger.info("Training ready")

logger.info("Serializing network setup to simple_example_config.txt")
with open("simple_example_config.txt", "w") as f:
    serialize_neural_network(neural_network, f)

logger.info("Testing")
print("Result Expected Diff")

good = 0
bad = 0
for _ in range(0, 20):
    p = np.random.random(2)
    result = neural_network.feedforward(p)
    result = 0 if result[0] < 0.5 else 1
    if result == team(p):
        good += 1
    else:
        bad += 1
    print(result, team(p), abs(result - team(p)))

print("Good: %d Bad: %d Success Rate: %f" %
      (good, bad, float(good) / float(good+bad)))

