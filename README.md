# NN-Autograd
This repository aims to replicate Andrej Karpathy's micrograd and MLP from scratch. It has been further extended by my own implementation of an optimizer class and a batch loader.

The core is the Node data structure that can compute gradients by using the chain rule and creating a computational graph.

The MLP class is based on the Layer class, which itself is based on the Neuron class. Each neuron has a Node to save its value.

By defining a loss function for the output of the MLP, one can backpropagate the gradient and use an optimizer (such as gradient descent) to optimize the parameters.

An optimizer abstract class is defined to be able to initialize different types of optimizers (currently only SGD), similar to Pytorch as the parameters are a mutable data type.

A batch loader class is also present that takes data and can split it into batches. The batch loader can be looped through, similar to DataLoader in Pytorch.

The main function includes a custom toy classification dataset that demonstrates the functionality of the above.