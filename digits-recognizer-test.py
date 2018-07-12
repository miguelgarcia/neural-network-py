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

from neuralnetwork.network import load_from_file
from mnist import load_dataset, result_to_digit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digits_recognizer")


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
