import numpy as np
from torch import nn as nn

from tac.agent.net import activation_functions
from tac.agent.net.base import Net
from tac.agent.net.net_util import build_fc_model, init_layers, get_loss_fn
from tac.utils.general import set_attr_from_dict

import pydash as ps


class MLPNet(Net, nn.Module):

    def __init__(self, net_spec, input_size, output_size):

        nn.Module.__init__(self)
        super().__init__(net_spec, input_size, output_size)

        # Set default attributes
        set_attr_from_dict(self, dict(
            out_layer_activation=None,
            init_fn=None,
            clip_grad_val=None,
            loss_spec={"name": "MSELoss"},
            optim_spec={"name": "adam"},
            lr_scheduler_spec=None,
            update_type="replace",
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        # Overwrite default values with any values from spec file
        set_attr_from_dict(self, net_spec, keys=[
            "shared",
            "hidden_layer_units",
            "hidden_layer_activation",
            "init_fn",
            "clip_grad_val",
            "lr_scheduler_spec",
            "update_type",
            "update_frequency",
            "polyak_coef",
            "gpu",
        ])

        # Model body (all except the output layer)
        dims = [self.in_dim] + self.hidden_layer_units    # List of units, including input and hidden layers
        self.model = build_fc_model(dims, activation=self.hidden_layer_activation)

        layers = []

        # Output layer (model tail).
        if isinstance(self.out_dim, (int, np.int64)):
            self.model_tail = build_fc_model([dims[-1], self.out_dim], activation=self.out_layer_activation)
        else:
            raise NotImplementedError("Only integer output dimensions are currently supported (single tail).")

        init_layers(self, self.init_fn)    # Initialise the weights of the layers of the network
        self.loss_fn = get_loss_fn(self.loss_spec)    # Get the loss function (e.g. MSELoss)
        self.to(self.device)   # Move the network to the GPU (if available)
        self.train()  # Set the module in training mode. Inherited from nn.Module. Relevant for dropout and BatchNorm

    def forward(self, x):
        x = self.model(x)
        x = self.model_tail(x)
        # TODO: hack to fix Torch double/float issue (
        #  https://stackoverflow.com/questions/67456368/pytorch-getting-runtimeerror-found-dtype-double-but-expected-float)
        x = x.double()
        return x
