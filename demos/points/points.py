import numpy as np
import matplotlib.pyplot as plt
from wingrad.tensor import Tensor
from wingrad.net import MLP
from sklearn.datasets import make_moons


# configure data
X, y = make_moons(n_samples=400, noise=0.08)

# make y either -1 or 1
y = y*2 - 1

# visualize the data
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')
plt.show()   

# reshape data
X = X.T
y = y.reshape(400, 1).T


# training examples and labels
inputs = [X[:, i].reshape(2, 1) for i in range(X.shape[1])]    # list of 100 inputs with shape (2, 1)
labels = [y[:, i].reshape(1, 1) for i in range(y.shape[1])]    # list of 100 labels with shape (1, 1)


# initialize a model
model = MLP(2, [16, 1])


# training loop
for k in range(200):

    epochs = 200

    # forward pass
    pred = [model(x) for x in inputs]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(labels, pred)])
    accuracy = [(yi > 0) == (predi.data > 0) for yi, predi in zip(labels, pred)]
    acc = sum(accuracy) / len(accuracy)

    # zero out gradients before backward pass
    model.zero_grad()

    # backward pass
    loss.backward()

    # update parameters
    for p in model.parameters():
        p.data -= 0.009 * p.grad

    # print the step, loss, and accuracy on that step 10 times througout training
    num_prints = 10 if epochs >= 10 else epochs
    if (k % (epochs // num_prints) == 0 or k == epochs-1):
        print(f"step {k}, loss: {loss.data[0][0]}, accuracy: {acc[0][0]*100}%")


# Visualize the decision boundary
h = 0.01  # Step size for the meshgrid
x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_mesh = np.c_[xx.ravel(), yy.ravel()]  # Flatten meshgrid coordinates
inputs = [Tensor(xrow).reshape((2, 1)) for xrow in X_mesh]
scores = [model(x).data > 0 for x in inputs]  # Predict scores for each point
Z = np.array(scores).reshape(xx.shape)


# Plot decision boundary and data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Paired, edgecolors='k')
plt.title('Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()