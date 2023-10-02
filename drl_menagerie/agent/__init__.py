from torch import nn as nn, optim as optim


activation_functions = {
    "relu": nn.ReLU,
}

optimisers = {
    "adam": optim.Adam,
}
