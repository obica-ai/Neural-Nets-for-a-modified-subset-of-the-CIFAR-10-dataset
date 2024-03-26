# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run,
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader, TensorDataset


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256

        We recommend setting lrate to 0.01 for part 1.

        """

        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            # Batch normalization normalizes the activations of the neurons
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # Dropout after activation function

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout after activation function

            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout after activation function
            nn.BatchNorm2d(128),

            # nn.Conv2d(128, 256, kernel_size=3, stride=2),
            # nn.ReLU(),
            # nn.Dropout(0.5),  # Dropout after activation function
            # nn.BatchNorm2d(256),
        )
        # mat 1 100*900
        # mat 2 2883*64
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1152, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=out_size),
            nn.BatchNorm1d(4),
            nn.ReLU(),

        )

        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=0.9, weight_decay=0.001)

    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        # print(x.shape)
        x = x.reshape(-1, 3, 31, 31)
        x = self.model(x)

        x = x.view(x.shape[0], -1)
        return self.classifier(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        # raise NotImplementedError("You need to write this part!")
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()
        x = loss.detach().cpu().numpy().item()
        return x


def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    """
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    mu = train_set.mean(dim=0)
    sig = train_set.std(dim=0)

    sig[sig == 0] = 1e-10

    train_set = (train_set - mu) / sig
    stan_dev = (dev_set - mu) / sig

    train_data = TensorDataset(train_set, train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    in_size = len(train_set[0])
    out_size = 4
    lrate = 0.01
    loss_fn = nn.CrossEntropyLoss()

    net = NeuralNet(lrate, loss_fn, in_size, out_size)
    losses = []
    yhats = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            loss = net.step(inputs, labels)
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)

        # 5. Evaluate on the development set
    net.eval()
    for img in stan_dev:
        y = torch.argmax(net.forward(img),dim=1).numpy()
        yhats.append(y.item())
    # with torch.no_grad():
    #     logits = net(stan_dev)
    #     yhats = torch.argmax(logits, dim=1).numpy()
    yhats = np.array(yhats)
    return losses, yhats, net
    # raise NotImplementedError("You need to write this part!")
    # return [],[],None
