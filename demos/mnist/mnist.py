import numpy as np
from wingrad.net import MLP
from keras.datasets import mnist


# load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()


# one-hot function, zeros are made into -1 to accomodate tanh function
def one_hot(Y, num_classes):
    oh = np.eye(num_classes)[Y.reshape(-1)]
    oh[oh == 0] = -1
    return oh


# Flatten, normalize, and reshape data for model
train_x = np.asarray(train_x.reshape(train_x.shape[0], -1) / 255).T    
test_x = np.asarray(test_x.reshape(test_x.shape[0], -1) / 255).T       
train_y = one_hot(train_y, num_classes=10).T                           
test_y = one_hot(test_y, num_classes=10).T                             


# initialize a model
model = MLP(784, [16, 16, 10])


# loss function
def loss(batch_size=None):

    # handle minibatches
    if batch_size is None:
        Xb, yb = train_x, train_y 
    else:
        batch = np.random.permutation(train_x.shape[1])[:batch_size]
        Xb, yb = train_x[:, batch], train_y[:, batch]

    # lists of individual training examples. len of the lists equal to batch_size.
    inputs = [Xb[:, i].reshape(784, 1) for i in range(Xb.shape[1])]
    labels = [yb[:, i].reshape(10, 1) for i in range(yb.shape[1])]

    # forward pass to get predictions
    pred = [model(x) for x in inputs]

    # get the losses
    loss = sum([(yout.sum(axis=0) - ygt.sum(axis=0))**2 for ygt, yout in zip(labels, pred)])

    # accuracy percentage
    accuracy = [(yi > 0) == (predi.data > 0) for yi, predi in zip(labels, pred)]

    return loss, sum(accuracy) / len(accuracy)


for k in range(500):

    epochs = 500

    # forward pass
    total_loss, acc = loss(batch_size=512)

    # zero grad
    model.zero_grad()

    # backward pass
    total_loss.backward()

    # update parameters using adaptive moment estimation (Adam)
    for p in model.parameters():
        p.data -= 0.0005 * p.grad 

    # print runtime performance updates
    num_prints = 10 if epochs >= 10 else epochs
    if (k % (epochs // num_prints) == 0 or k == epochs-1):
        print(f"step {k}, loss: {total_loss.data[0]:.5f}, accuracy: {acc[0][0]*100:.3f}%")

