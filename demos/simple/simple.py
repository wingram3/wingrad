import numpy as np
from wingrad.net import MLP

# initialize model
m = MLP(3, [9, 9, 1])

# simple training features
xs = np.array([[2.0, 3.0, 0.5, 1.0],
               [3.0, -1.0, 1.0, 1.0],
               [-1.0, 0.5, 1.0, -1.0]])

# simple training labels
ys = np.array([[1.0], [-1.0], [-1.0], [1.0]])

# reshaping and configuring the data for model
dataset = [xs[:, i].reshape(3, 1) for i in range(4)]

# training loop
for k in range(30):

    # forward pass
    ypred = [m(d) for d in dataset]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

    # zero grad
    m.zero_grad() 

    # backward pass
    loss.backward()

    # update parameters
    for p in m.parameters():
        p.data += -0.09 * p.grad

    print(k, loss.data, ypred)
