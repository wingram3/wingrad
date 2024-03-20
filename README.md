# wingrad - a vectorized translation of Andrej Karpathy's 'micrograd'

See the original micrograd here: https://github.com/karpathy/micrograd

Wingrad, like micrograd, is a small autograd engine that implements backpropagation over a directed acylcic graph (DAG). It allows for the construction of multilayer perceptrons out of Tensor objects, which are detailed more below.

## How it works

A Tensor, as defined in the file '''tensor.py''', is essentially an array of values on which operations like addition, subtraction, element-wise multiplication, dot multiplication, activation functions, etc. can be called. Each Tensor keeps track of the other Tensors that made it my some operation. As a computation graph is built up from subsequent operations of tensors on one another, a call of the backward() method on a Tensor will create a topologically sorted list of Tensors that came before it. Going through that list in reverse order, it will find the derivatives of the last Tensor in the graph with respect to all other tensors that came before it. This is an application of the chain rule from calculus, and is known in machine learning as reverse-mode automatic differentiation.

In neural networks, and specifically for supervised deep learning problems, reverse-mode autodiff is useful because it allows for the adjustment of network parameters based on the negative gradient of a loss function, also known as gradient descent optimization. The loss function is computed after a forward pass through the network, which is a function of how close the network was to predicting the right value or set of values. By finding the derivatives of the loss function with respect to each individual network parameter, it is possible to improve the network's performance by nudging the parameter values either up or down to drive the loss function to zero. 

## Differences between wingrad and micrograd

### The use of Tensors vs. Values
While wingrad and micrograd both allow for the construction of nueral nets using DAGs and reverse-mode autodiff, wingrad is slightly more complex in its implementation because of its use of array-like Tensor objects, as opposed to Value objects, which hold single scalar values. In neural nets, Tensors allow us to take advantage of the parallelized structure of GPUs and CPUs, leading to faster calculations of layer activations and gradients of parameters. All if this means we are able to train neural network models much faster. This kind of defeats the purpose of micrograd as a teaching tool, but it was a fun challenge to implement Tensors into wingrad.

## Non-recursive backward() function


## No Neuron class in wingrad
