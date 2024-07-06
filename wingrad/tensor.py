import numpy as np
from collections import deque
from typing import Union, Tuple, Callable, Set

class Tensor:
    """
    A tensor, as defined in this file, is basically an array of values on which operations
    like add, multiply, etc. can be called, as well as a '_backward()' function.
    The _backward() function will find the derivatives of the child tensors w.r.t.
    the tensor on which it's called, which also has a derivative w.r.t. some tensor after it.
    Essentially, _backward() locally applies the chain rule from calculus.
    """

    def __init__(self, data: Union[np.ndarray, list], _children: Tuple['Tensor', ...] = (), _op: str = '') -> None:
        """
        Initialize a tensor.

        Args:
            data: The data for the tensor. Can be a numpy array or a list.
            _children: A tuple of child tensors that contributed to this tensor.
            _op: A string representing the operation that created the tensor.
        """
        data = data if isinstance(data, np.ndarray) else np.array(data)  # turn data into numpy array if it isn't one already
        self.data: np.ndarray = data                        # the tensor's data
        self.grad: np.ndarray = np.zeros_like(data)         # gradient/derivative of the tensor w.r.t. whatever tensor _backward() was called on
        self._backward: Callable[[], None] = lambda: None   # _backward() function, depends on what operation made the tensor
        self._prev: Set = set(_children)          # set of the two tensors that made the tensor by some operation
        self._op: str = _op                                 # operation that made the tensor from its child tensors
        self.shape: Tuple[int, ...] = self.data.shape       # dimensions of the tensor's data

    def __add__(self, other) -> 'Tensor':
        """
        Add two tensors.

        Args:
            other: The tensor to add.

        Returns:
            A new tensor resulting from the addition.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)  # turn other into a tensor if it isn't one already
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward() -> None:
            self.grad += out.grad.astype(self.grad.dtype).reshape(self.shape)
            other.grad += out.grad.astype(other.grad.dtype).reshape(other.shape)
        out._backward = _backward

        return out

    def __mul__(self, other) -> 'Tensor':
        """
        Multiply two tensors (element-wise).

        Args:
            other: The tensor to multiply.

        Returns:
            A new tensor resulting from the multiplication.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __xor__(self, other) -> 'Tensor':
        """
        Multiply two tensors (dot product).

        Args:
            other: The tensor to perform the dot product with.

        Returns:
            A new tensor resulting from the dot product.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other), 'dot')

        def _backward() -> None:
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward

        return out

    def __pow__(self, other: Union[float, int]) -> 'Tensor':
        """
        Raise a tensor to the power of a float or int.

        Args:
            other: The exponent, which must be a float or int.

        Returns:
            A new tensor resulting from the exponentiation.
        """
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward() -> None:
            self.grad += (other * self.data**(other - 1.0)) * out.grad
        out._backward = _backward

        return out

    def exp(self) -> 'Tensor':
        """
        Compute the exponential of each element in a tensor.

        Returns:
            A new tensor with the exponential of each element.
        """
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward() -> None:
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self) -> 'Tensor':
        """
        Compute the natural log of each element in a tensor.

        Returns:
            A new tensor with the natural log of each element.
        """
        out = Tensor(np.log(self.data), (self,), 'ln')

        def _backward() -> None:
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out

    def tanh(self) -> 'Tensor':
        """
        Compute the tanh activation function.

        Returns:
            A new tensor with the tanh of each element.
        """
        x = self.data
        t = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        out = Tensor(t, (self,), 'tanh')

        def _backward() -> None:
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def relu(self) -> 'Tensor':
        """
        Compute the ReLU activation function.

        Returns:
            A new tensor with the ReLU of each element.
        """
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward() -> None:
            self.grad += (np.where(out.data <= 0, 0, 1)) * out.grad
        out._backward = _backward

        return out

    def backward(self) -> None:
        """ Autograd function to compute gradients. """
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

    def __neg__(self) -> 'Tensor':
        """
        Negation (for subtraction).

        Returns:
            A new tensor with negated elements.
        """
        return self * -1

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Subtraction (uses existing __add__() method).

        Args:
            other: The tensor to subtract.

        Returns:
            A new tensor resulting from the subtraction.
        """
        return self + (-other)

    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Reverse add. In case an operation is called on a non-tensor.

        Args:
            other: The tensor or value to add.

        Returns:
            A new tensor resulting from the addition.
        """
        return self + other

    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Reverse subtract.

        Args:
            other: The tensor or value to subtract from.

        Returns:
            A new tensor resulting from the subtraction.
        """
        return other + (-self)

    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Reverse multiply.

        Args:
            other: The tensor or value to multiply.

        Returns:
            A new tensor resulting from the multiplication.
        """
        return self * other

    def __rxor__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Reverse multiply (dot product).

        Args:
            other: The tensor or value to multiply (dot product).

        Returns:
            A new tensor resulting from the dot product.
        """
        return self ^ other

    def __repr__(self) -> str:
        """
        String representation of the tensor.

        Returns:
            A string representation of the tensor's value.
        """
        return f'Tensor(data={self.data})'

    def sum(self, axis: int = 0) -> 'Tensor':
        """
        Sum all elements in a tensor along a specified axis.

        Args:
            axis: The axis along which to sum.

        Returns:
            A new tensor with summed elements.
        """
        self.data = np.sum(self.data, axis=axis)
        self.grad = np.sum(self.grad, axis=axis)
        self.shape = self.data.shape
        return self

    def reshape(self, newshape: Tuple[int, ...]) -> 'Tensor':
        """
        Reshape a tensor's data.

        Args:
            newshape: The new shape for the tensor.

        Returns:
            A new tensor with reshaped data.
        """
        self.data = self.data.reshape(newshape)
        self.grad = self.grad.reshape(newshape)
        self.shape = self.data.shape
        return self

    def transpose(self) -> 'Tensor':
        """
        Transpose a tensor's data.

        Returns:
            A new tensor with transposed data.
        """
        self.data = self.data.T
        self.grad = self.grad.T
        self.shape = self.data.shape
        return self
