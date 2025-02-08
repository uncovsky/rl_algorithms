from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

def get_activation(act_str):
    # Map activation strings to activation functions
    act_str = act_str.lower()
    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=-1),
        "leakyrelu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "id": nn.Identity()  # No activation
    }

    if act_str not in activations:
        raise ValueError(f"Unsupported activation: {act_str}")

    return activations[act_str]

# desc format - triples (activation, input, output)
def make_mlp(net_desc):
    d = OrderedDict()
    for i, (activation, input_size, output_size) in enumerate(net_desc):

        d[f"linear_{i}"] = nn.Linear(input_size, output_size)
        d[f"{activation.lower()}_{i}"] = get_activation(activation)
    return nn.Sequential(d)

def initialize_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
