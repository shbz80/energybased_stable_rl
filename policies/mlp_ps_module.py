"""MLP Module."""

from torch.nn import functional as F
import torch
import torch.nn as nn
from garage.torch import NonLinearity

class LinearList(nn.Module):
    def __init__(self, input_dim, output_dim, col=True):
        super(LinearList, self).__init__()
        self.p_list = nn.ParameterList()
        self.col = col
        for i in range(input_dim if col else output_dim):
            if col:
                p = nn.Parameter(torch.Tensor(output_dim , 1))
            else:
                p = nn.Parameter(torch.Tensor(1, input_dim))
            self.p_list.append(p)
        self.bias = nn.Parameter(torch.Tensor(output_dim))

    def init(self, w_init, b_init):
        for p in self.p_list:
            w_init(p)
        b_init(self.bias)

    def forward(self, input):
        weights = [p for p in self.p_list]
        if self.col:
            weight = torch.cat(weights, dim=1)
        else:
            weight = torch.cat(weights, dim=0)
        return nn.functional.linear(input, weight, self.bias)


class MLPPSModule(nn.Module):
    """MLP Model.

    A Pytorch module composed only of a multi-layer perceptron (MLP), which
    maps real-valued inputs to real-valued outputs.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_):
        super().__init__()

        self._output_dim = output_dim

        self._layers = nn.ModuleList()

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            linear_layer = LinearList(prev_size, size, col=True)
            linear_layer.init(hidden_w_init, hidden_b_init)
            hidden_layers.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))

            self._layers.append(hidden_layers)
            prev_size = size

        self._output_layer = nn.ModuleList()
        output_layer = nn.Sequential()
        linear_layer = LinearList(prev_size, output_dim, col=False) # rows instead of columns only to reduce p_list
        linear_layer.init(output_w_init, output_b_init)
        output_layer.add_module('linear', linear_layer)

        if output_nonlinearity:
            output_layer.add_module('non_linearity',
                                    NonLinearity(output_nonlinearity))

        self._output_layer.append(output_layer)

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            torch.Tensor: Output value

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return self._output_layer[0](x)

    @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dim
