import torch
import torch.nn as nn
from functools import partial

from tac.utils.general import get_class_name

import pydash as ps


def build_fc_model(dims, activation=None):
    """
    Builds a fully-connected model, by interleaving nn.Linear with any accompanying activation functions.

    Parameters
    ----------
    dims : list of int
        The dimensions of the model. Each element of the list specifies the number of units in that layer.
    activation : nn.Module
        The activation function to use between each layer. If None, no activation function is used.

    Returns
    -------
    model : nn.Sequential
        The fully-connected model
    """
    assert len(dims) >= 2, "Dims need to contain at least an input layer and an output layer"
    # Make pairs of (in, out) dims per layer
    dim_pairs = list(zip(dims[:-1], dims[1:]))
    layers = []
    for in_d, out_d in dim_pairs:
        layers.append(nn.Linear(in_d, out_d))
        if activation is not None:
            layers.append(get_activation_fn(activation))
    model = nn.Sequential(*layers)
    return model


def get_nn_name(uncased_name):
    """Helper, to get the proper name in PyTorch nn, given a case-insensitive name"""
    for nn_name in nn.__dict__:
        if nn_name.lower() == uncased_name.lower():
            return nn_name
    raise ValueError(f"Name {uncased_name} not found in {nn.__dict__}")


def init_layers(net, init_fn_name):
    """
    Primary method to initialise the weights of the layers of a network.

    Parameters
    ----------
    net : nn.Module
        The network whose layers are to be initialised. This function is used within the method of the network class,
        and the network class itself is passed as the net argument.
    init_fn_name : str
        The name of the initialisation function to use. This is the name of a function in the torch.nn.init library.

    """
    if init_fn_name is None:
        return

    # Get nonlinearity name
    nonlinearity = get_nn_name(net.hidden_layer_activation).lower()

    # Get initialisation function
    init_fn = getattr(nn.init, init_fn_name)

    # Conditionally, add arguments depending on the nonlinearity (this block not yet implemented)
    problem_activation_substrings = {"kaiming", "orthogonal", "xavier"}
    for problem_substring in problem_activation_substrings:
        if problem_substring in init_fn_name.lower():
            raise NotImplementedError(f"Initialisation function {init_fn_name} not supported for activation")

    # Apply init params to each layer of the network.

    # MLP inherits from nn.Module, which contains the method named "apply", which applies a function recursively to
    # every submodule (as returned by the .children() method) as well as the module itself.

    # partial() is a function from functools, which allows you to create a new function from an existing function,
    # but with some arguments already filled in. Here, we create a new function called init_params, which is the same
    # as the init_params function defined below, but with the init_fn argument already filled in.

    # This is necessary, as nn.Module.apply() requires a function with a single argument, but init_params requires
    # two. apply is called recursively.

    net.apply(partial(init_params, init_fn=init_fn))


def init_params(module, init_fn):
    """Initialise module's weights using init_fn, and biases to 0.0"""
    bias_init = 0.0
    classname = get_class_name(module)
    if "Net" in classname:
        # Skip if it's a net (as opposed to a pytorch layer)
        pass
    elif classname == "BatchNorm2d":
        # Skip - can't init BatchNorm2d with weight and bias
        pass
    elif any(k in classname for k in ("Conv", "Linear")):
        init_fn(module.weight)    # Initialise weights
        nn.init.constant_(module.bias, bias_init)    # Initialise biases
    elif "GRU" in classname:
        raise NotImplementedError("GRU not yet supported")
    else:
        raise NotImplementedError(f"Class {classname} not yet supported")


def get_loss_fn(loss_spec):
    """Helper to parse loss param and construct loss_fn for the net"""
    loss_class = getattr(nn, get_nn_name(loss_spec["name"]))
    loss_spec = ps.omit(loss_spec, "name")    # Remove the name key from the loss_spec dict
    loss_fn = loss_class(**loss_spec)
    return loss_fn


def get_activation_fn(activation):
    """Helper to get the activation function for the net, from the `activation` string"""
    activation_class = getattr(nn, get_nn_name(activation))
    activation_fn = activation_class()
    return activation_fn
