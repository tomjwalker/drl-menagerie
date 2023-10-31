from torch import nn as nn

from tac.agent.net import activation_functions


class MLPNet(nn.Module):

    def __init__(self, spec_dict, input_size, output_size):
        super(MLPNet, self).__init__()

        layers = []

        num_hidden_layers = len(spec_dict["hidden_layer_units"])

        # Input layer
        layers.append(nn.Linear(input_size, spec_dict["hidden_layer_units"][0]))
        layers.append(activation_functions.get(spec_dict["activation"])())

        # Hidden layers
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(spec_dict["hidden_layer_units"][i], spec_dict["hidden_layer_units"][i + 1]))
            layers.append(activation_functions.get(spec_dict["activation"])())

        # Output layer
        layers.append(nn.Linear(spec_dict["hidden_layer_units"][-1], output_size))

        self.model = nn.Sequential(*layers)

        self.train()  # Set the module in training mode. Inherited from nn.Module. Relevant for dropout and BatchNorm

    def forward(self, x):
        return self.model(x)

    def train_step(self, loss, optimiser):
        """
        Performs a single training step. This is a backward pass, followed by a gradient update.

        Arguments:
        ----------
        loss: torch.Tensor
            The loss to be minimised
        optimiser: torch.optim.Optimiser
            The optimiser to use for the gradient update
        """
        # optimiser.zero_grad() clears the gradients of all optimisable variables, from the previous time step
        optimiser.zero_grad()
        # loss.backward() computes the gradient of the loss w.r.t. all optimisable variables, del(loss)/del(theta)
        loss.backward()
        # optimiser.step() updates the value of the optimisable variables, using the gradients computed in the previous
        # step
        optimiser.step()
        return loss
