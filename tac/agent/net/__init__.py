from torch import nn as nn, optim as optim


#######################################################################################################################
# Name mappings
#######################################################################################################################

activation_functions = {
    "relu": nn.ReLU,
}
optimisers = {
    "adam": optim.Adam,
}
