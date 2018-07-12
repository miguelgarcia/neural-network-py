# neural-network-py
A configurable N layers neural network implemented in Python3. Trainer code uses back propagation.

## Install dependencies
The only dependency is NumPy, install it running:

`pip3 install -r requirements.txt`

## Defining a neural network
Example to create a network of three layers with 784 neurons in the first layer, 20 in the second and 10 in the last one.

```python
from neuralnetwork.network import NeuralNetwork

neural_network = NeuralNetwork([784, 20, 10])
```

## Training a neural network

```python
from neuralnetwork.network import NeuralNetwork
from neuralnetwork.training import Trainer, TrainingDataSample

# training_data is a list of TrainingDataSample
# Each TrainingDataSample has the input value and its expected result
# See simple-example.py and digits-recognizer-traini.py

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

```

This will train the network using the stocastic gradient descent method

### Algorithm
0. Build a batch used for cost estimation, if `cost_estimator_batch_size`
    is `None`, use all the samples in `training_data`.
1. Estimate current cost for cost estimation batch
2. Pick a batch of `batch_size` samples from `training_data`
3. Repeat `batch_iterations`:
    1. Calculate cost function gradient for each sample
    2. Average the gradients
    3. Scale gradients by `learning_step`
    4. Update networks weights and biases substracting the gradients
4. Estimate updated cost for cost estimation batch
5. The picked batch produced an "improvement" if 
    `updated cost < min(previous costs) * (1-min_improvement_per_batch)`
6. Repeat until `max_batches_without_improvement ` consecutive batches are
    processed without making improvement (as defined in 5).

## Using the neural network to process data

```python
result = neural_network.feedforward(data)
```

## Saving and loading trained networks

Since training can require a lot of time it's useful to be able to save and restore neural networks parameters.

* Saving network configuration:
```python
from neuralnetwork.network import NeuralNetwork, serialize_neural_network

with open("network_setup.txt", "w") as f:
    serialize_neural_network(neural_network, f)

```

* Loading network configuration:
```python
from neuralnetwork.network import load_from_file

with open("network_setup.txt", "r") as f:
    neural_network = load_from_file(f)
```

# Working example

## Simple example

See example `simple-example.py`. A points classifier in two teams depending on if `x > y`.

## Hand written digits recognizer

This example trains and uses a neural network to recognize hand written digits.

I used the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset to train and test the network. The dataset has 60000 digits for training and 10000 for testing. This example needs its inputs in CSV format, run the script in `datasets/mnist/download_datasets.sh` to get them.

Since the inputs are 28x28 pixels = 784 pixels and the outputs are 10 possible digits the network must have 784 neurons in the first layer and 10 in the last one. I choose to have only one intermediate layer of 20 neurons.

* `digits-recognizer-train.py` trains the network and produces `digits_recognizer_setup.txt` with the trained network configuration.

* `digits-recognizer-test.py` test each case in the testing dataset and outputs the success rate.

This example achieves a success rate of ~93%, that could be improved re training and trying a different number of hidden layers and neurons count in each layer.

# Theorical background and references
I've used many sources of information and really recommend

- Videos from 3Blue1Brown 

    - [But what *is* a Neural Network? | Chapter 1, deep learning](https://www.youtube.com/watch?v=aircAruvnKk)
    - [Gradient descent, how neural networks learn | Chapter 2, deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w)
    - [What is backpropagation really doing? | Chapter 3, deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
    - [Backpropagation calculus | Appendix to deep learning chapter 3](https://www.youtube.com/watch?v=tIeHLnjs5U8)

 - Neural Networks and Deep Learning free online book, specially chapter 2. [Book site](http://neuralnetworksanddeeplearning.com)


