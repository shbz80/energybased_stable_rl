import numpy as np
from garage.torch import NonLinearity

import torch
import torch.nn as nn

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())


class ICNN(nn.Module):
    # check if all params are registered todo
    def __init__(self,
                 input_dim,
                 hidden_sizes,
                 w_init=nn.init.xavier_normal_,
                 b_init=nn.init.zeros_,
                 nonlinearity=torch.relu,
                 ):
        super(ICNN, self).__init__()

        self._y_layers = nn.ModuleList()
        for size in hidden_sizes:
            linear_layer = nn.Linear(input_dim, size)
            w_init(linear_layer.weight)
            b_init(linear_layer.bias) # check if bias is there todo
            self._y_layers.append(linear_layer)
        linear_layer = nn.Linear(input_dim, 1)
        w_init(linear_layer.weight)
        b_init(linear_layer.bias)
        self._y_layers.append(linear_layer)

        self._z_layers = nn.ModuleList()
        prev_size = hidden_sizes[0]
        for size in hidden_sizes[1:]:
            positive_linear_layer = PositiveLinear(prev_size, size)
            w_init(positive_linear_layer.log_weight)
            self._z_layers.append(positive_linear_layer)
            prev_size = size
        positive_linear_layer = PositiveLinear(prev_size, 1)
        w_init(positive_linear_layer.log_weight)
        self._z_layers.append(positive_linear_layer)

        self.nonlinearity = NonLinearity(nonlinearity)

    def forward(self, z):

        z0 = z.clone()

        z = self.nonlinearity(self._y_layers[0](z0))

        layer_num = len(self._z_layers)
        for i in range(layer_num):
            z = self.nonlinearity(self._y_layers[i+1](z0) + self._z_layers[i](z))
        return z

class Damping(nn.Module):
    def __init__(self,
                 input_dim,  # input should be velocity
                 hidden_sizes,
                 w_init=nn.init.xavier_normal_,
                 b_init=nn.init.zeros_,
                 hidden_nonlinearity=torch.tanh,
                 full_mat=True,
                 ):
        super(Damping, self).__init__()

        N = input_dim
        self.offdiag_output_dim = N*(N-1)//2
        self.diag_output_dim = N
        self.output_dim = self.offdiag_output_dim + self.diag_output_dim
        self.full_mat = full_mat

        self._layers = nn.ModuleList()

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            linear_layer = nn.Linear(prev_size, size)
            w_init(linear_layer.weight)
            b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)
            hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))
            self._layers.append(hidden_layers)
            prev_size = size

        if self.full_mat:
            out_dim = self.output_dim
        else:
            out_dim = self.diag_output_dim

        self._output_layer = nn.ModuleList()
        linear_layer = nn.Linear(prev_size, out_dim)
        w_init(linear_layer.weight)
        b_init(linear_layer.bias)
        self._output_layer.append(linear_layer)

    def forward(self, input):

        x = input.view(1,-1)
        for layer in self._layers:
            x = layer(x)

        x = self._output_layer[0](x)
        n = self.diag_output_dim
        diag_idx = np.diag_indices(n)
        D = torch.zeros(x.shape[0], n, n)

        for i in range(x.shape[0]):
            L = torch.zeros(n, n)
            if self.full_mat:
                diag_elements = torch.exp(x[i, :self.diag_output_dim])
                off_diag_elements = x[i, self.diag_output_dim:]
                off_diag_idx = np.tril_indices(n, k=-1)
                L[off_diag_idx] = off_diag_elements
                L[diag_idx] = diag_elements
                D[i] = L@L.t()
            else:
                diag_elements = torch.exp(x[i])
                D[i][diag_idx] = diag_elements

        return D


