from torch import nn as nn, optim as optim
from .algorithm.reinforce import Reinforce

algorithms = {
    "reinforce": Reinforce,
}

activation_functions = {
    "relu": nn.ReLU,
}
optimisers = {
    "adam": optim.Adam,
}
