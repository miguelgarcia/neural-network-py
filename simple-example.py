#!/bin/env python3
# Mini example, classify points in two teams:
# "team 1" if x > y, "team 0" otherwise
import numpy as np

from neuralnetwork import NeuralNetwork
from training import Trainer, TrainingDataSample


# Use a network with two layers:
# Input has 2 neurons
# Output has 1 neuron
n = NeuralNetwork([2, 1])

# Calculate the correct result for a point
def team(p):
    if p[0] > p[1]:
        return 1
    return 0


print("Generating training data")
training_data = []
for _ in range(0, 100):
    p = np.random.random(2)
    training_data.append(TrainingDataSample(p, np.array([team(p)])))


print("Training")
t = Trainer()
t.train(n, training_data, 0.5, 1000, 5)

print("Training ready")

print("Testing:")
print("Result Expected Diff")
for _ in range(0, 20):
    p = np.random.random(2)
    result = n.feedforward(p)
    result = 0 if result[0] < 0.5 else 1
    print(result, team(p), abs(result - team(p)))

# Note: since neural networks are randomly initialized it will be better to
# Create and train N networks and then choose the best one.
