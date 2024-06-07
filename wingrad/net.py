from wingrad.tensor import Tensor
import numpy as np

"""
This file contains the two modules needed to construct neural nets out of tensor objects:
    - Layer module: a layer of neurons, where neurons are represented by their weights, biases, and activations
    - Multi-layer Perceptron module: builds a neural net with specified layer sizes out of Layer objects
"""


class Module:
    """ Base class for all modules. """

    def zero_grad(self):
        """ Zero out gradients before backpropagation. """
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        """
        Return the parameters of the module.

        Returns:
            list: An empty list of parameters.
        """
        return []


class Layer(Module):
    """ A layer of neurons. """

    def __init__(self, nin, nout):
        """
        Initialize the layer with random weights and zero biases.

        Args:
            nin (int): Number of input neurons.
            nout (int): Number of output neurons.
        """
        self.w = Tensor(np.random.randn(nout, nin))  # tensor of all weights in a layer
        self.b = Tensor(np.zeros((nout, 1)))  # tensor of all biases in a layer

    def __call__(self, x):
        """
        Forward pass through the layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Activated output tensor.
        """
        wx = self.w ^ x  # weighted sum of weights and inputs (dot product)
        z = wx + self.b  # raw neuron activations
        a = z.tanh()  # z fed through activation function
        return a

    def parameters(self):
        """
        Return the parameters of the layer.

        Returns:
            list: List containing the weight and bias tensors.
        """
        return [self.w] + [self.b]


class MLP(Module):
    """ Multi-layer Perceptron module. """

    def __init__(self, nin, nouts):
        """
        Initialize the MLP with a list of Layer objects of specified sizes.

        Args:
            nin (int): Number of input neurons.
            nouts (list): List of integers specifying the number of neurons in each subsequent layer.
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Return all parameters in the MLP.

        Returns:
            list: List of all parameters in the MLP.
        """
        return [p for l in self.layers for p in l.parameters()]
