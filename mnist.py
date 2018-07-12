# Utility functions to load and transform MNIST datasets

import numpy as np
from neuralnetwork.training import TrainingDataSample

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