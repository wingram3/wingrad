import numpy as np
from collections import deque


class Tensor:

    """
    A tensor, as defined in this file, is basically an array of values on which operations like add, multiply, etc. can be called, 
    as well as a '_backward()' function. The _backward() function will find the derivatives of the child tensors w.r.t. the tensor on 
    which it's called, which also has a derivative w.r.t. some tensor after it. Essentially, _backward() locally applies the chain rule from calculus. 
    """
    
    def __init__(self, data, _children=(), _op=''):
        data = data if isinstance(data, np.ndarray) else np.array(data)    # turn data into numpy array if it isn't one already
        self.data = data                        # the tensor's data
        self.grad = np.zeros_like(data)         # gradient/derivative of the tensor w.r.t. whatever tensor _backward() was called on
        self._backward = lambda: None           # _backward() function, depends on what operation made the tensor
        self._prev = set(_children)             # set of the two tensors that made the tensor by some operation
        self._op = _op                          # operation that made the tensor from its child tensors
        self.shape = self.data.shape            # dimensions of the tensor's data

    # add two tensors
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)   # turn other into a tensor if it isn't one already
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad.astype(self.grad.dtype).reshape(self.shape)
            other.grad += out.grad.astype(other.grad.dtype).reshape(other.shape)
        out._backward = _backward

        return out
    
    # multiply two tensors (element-wise)
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    # multiply two tensors (dot product)
    def __xor__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other), 'dot')

        def _backward():
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward
        
        return out
    
    # take a tensor to the power of a float or int
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other - 1.0)) * out.grad
        out._backward = _backward 

        return out
    
    # exponential of each element in a tensor
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    # natural log of each element in a tensor
    def log(self):
        out = Tensor(np.log(self.data), (self,), 'ln')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out

    # tanh activation method
    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    # relu activation function
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (np.where(out.data <= 0, 0, 1)) * out.grad
        out._backward = _backward

        return out
    
    # autograd function
    def backward(self):

        topo = []
        visited = set()
        stack = deque([self])

        # topologically sort all tensors in computation graph in reverse order
        while stack:
            node = stack.popleft()
            if node not in visited:
                visited.add(node)
                stack.extend(node._prev)
                topo.append(node)
        
        # go through all tensors and apply _backward() function to get gradients
        self.grad = np.ones_like(self.data)
        for tensor in topo:
            tensor._backward()
    
    # negation (for subtraction)
    def __neg__(self):
        return self * -1
    
    # subtraction (uses existing __add__() method)
    def __sub__(self, other):
        return self + (-other)
    
    # reverse operation methods, in case an op. is called on a non-tensor and a tensor
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __rxor__(self, other):
        return self ^ other
    
    # nice representation of the tensor's value
    def __repr__(self):
        return f'Tensor(data={self.data})'
    
    # sum all elements in a tensor along an axis
    def sum(self, axis=0):
        self.data = np.sum(self.data, axis=axis)
        self.grad = np.sum(self.grad, axis=axis)
        self.shape = self.data.shape
        return self
    
    # reshape a tensor's data
    def reshape(self, newshape):
        self.data = self.data.reshape(newshape)
        self.grad = self.grad.reshape(newshape)
        self.shape = self.data.shape
        return self
    
    # transpose a tensor's data
    def transpose(self):
        self.data = self.data.T
        self.grad = self.grad.T
        self.shape = self.data.shape
        return self