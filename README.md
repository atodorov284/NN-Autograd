# NN-Autograd
This repository aims to replicate Andrej Karpathy's micrograd and MLP from scratch.

It has a Node data structure that can compute gradients by using the chain rule and creating a computational graph.

The MLP class is based on the Layer class, which itself is based on the Neuron class. Each neuron has a Node to save its value.

By defining a loss function for the output of the MLP, one can backpropagate the gradient and use an optimizer (such as gradient descent) to optimize the parameters.

All of the code is done from scratch and the only requirement is typing (for typehints). The only other library used is math.