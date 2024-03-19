from wingrad.tensor import Tensor
import numpy as np

'''
    This file contains the two modules needed to construct neural nets out of tensor objects:
        Layer module: a layer of neurons, where neurons are represented by their weights, biases, and activations
        Multi-layer Perceptron module: builds a neural net with specified layer sizes out of Layer objects
'''

class Module:

    # zero grad function - zeroes out gradients before backpropagation
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []


# a layer of neurons
class Layer(Module):

    def __init__(self, nin, nout):
        self.w = Tensor(np.random.randn(nout, nin))     # tensor of all weights in a layer
        self.b = Tensor(np.zeros((nout, 1)))            # tensor of all biases in a layer

    def __call__(self, x):
        wx = self.w ^ x     # weighted sum of weights and inputs (dot product)
        z = wx + self.b     # raw neuron activations
        a = z.tanh()        # z fed through activation function
        return a
    
    # parameters of a layer
    def parameters(self):
        return [self.w] + [self.b]


# multi-layer perceptron module
class MLP(Module):

    # initialized as a list of Layer objects of a specified size
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    # forward pass through the MLP
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    # all parameters in a MLP
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]