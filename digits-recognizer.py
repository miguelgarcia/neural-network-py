#!/usr/bin/env python3
# Mini example, classify points in two teams:
# "team 1" if x > y, "team 0" otherwise
import numpy as np
import sys
import os

from neuralnetwork import NeuralNetwork
from training import Trainer, TrainingDataSample

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

results_table = np.identity(10)

def load_dataset(path):
    samples = []
    with open(path, 'r') as f:
        for line in f:
            data=list(map(float, line.strip().split(',')))
            inp = np.array(data[1:]) / 255
            r=results_table[int(data[0])]
            samples.append(TrainingDataSample(inp, r))
    return samples

def result_to_digit(output):
    return np.argmax(output)

n = NeuralNetwork([784, 32, 10])

print("Training")
t = Trainer()
print("Loading training dateset")
training_data = load_dataset('./datasets/mnist/mnist_train.csv')
#training_data = load_dataset('./datasets/mnist/mnist_test.csv')
print("Training dateset loaded")
print("Training network")

# Perceptron 20
# Good: 97%
# learning_step=0.25
# batch_size=50
# max_iter=10
# min_improvement=0.000001
# max_batches_without_improvement = 100
# cost_estimator_batch_size = 5000

learning_step=0.15
batch_size=50
max_iter=10
min_improvement=0.000001
max_batches_without_improvement = 100
cost_estimator_batch_size = 6000
t.train(n, training_data, learning_step, max_iter, batch_size, min_improvement,
    max_batches_without_improvement, cost_estimator_batch_size, True)
del training_data
print("Trained")
#sys.exit(1)
print("Testing")
print("Loading training dateset")
testing_data = load_dataset('./datasets/mnist/mnist_test.csv')
print("Testing dateset loaded")
print("Testing network")
good = 0
bad = 0
for sample in testing_data:
    result = n.feedforward(sample.input)
    expected_result = result_to_digit(sample.output)
    actual_result = result_to_digit(result)
    print("%d %d %d" % (actual_result, expected_result, actual_result==expected_result))
    if actual_result==expected_result:
        good += 1
    else:
        bad +=1

print("Ended, good=%d, bad=%d" % (good, bad))