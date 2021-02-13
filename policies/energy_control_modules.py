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
        return nn.functional.linear(input, self.log_weight.exp(),bias=torch.zeros(self.out_features))       # todo try with bias = None


class ICNN(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_sizes,
                 w_init_y = nn.init.xavier_uniform_,
                 b_init_y = nn.init.zeros_,
                 w_init_z = nn.init.constant_,
                 w_init_z_param = 0.1,
                 nonlinearity=torch.relu,
                 ):
        super(ICNN, self).__init__()

        if len(hidden_sizes)<2:
            print('Requires at least 2 layers')
            assert(False)

        self._y_layers = nn.ModuleList()
        for size in hidden_sizes:
            linear_layer = nn.Linear(input_dim, size)
            w_init_y(linear_layer.weight)
            b_init_y(linear_layer.bias)
            self._y_layers.append(linear_layer)

        linear_layer = nn.Linear(input_dim, 1)
        w_init_y(linear_layer.weight)
        b_init_y(linear_layer.bias)
        self._y_layers.append(linear_layer)

        self._z_layers = nn.ModuleList()
        prev_size = hidden_sizes[0]
        for size in hidden_sizes[1:]:
            positive_linear_layer = PositiveLinear(prev_size, size)
            w_init_z(positive_linear_layer.log_weight, w_init_z_param)
            self._z_layers.append(positive_linear_layer)
            prev_size = size

        positive_linear_layer = PositiveLinear(prev_size, 1)
        w_init_z(positive_linear_layer.log_weight, w_init_z_param)
        self._z_layers.append(positive_linear_layer)

        self.nonlinearity = NonLinearity(nonlinearity)      # todo try without NonLinearity
        print('icnn init')

    def forward(self, z):

        z0 = z.clone()
        z = self.nonlinearity(self._y_layers[0](z0))

        layer_num = len(self._z_layers)
        for i in range(layer_num):
            z = self.nonlinearity(self._y_layers[i+1](z0) + self._z_layers[i](z))
        return z

    def relu_grad(self, z):
        z_ = z.clone()
        # z_.squeeze_()
        z_[(z_ > 0.0)] = 1.0
        z_[(z_ <= 0.0)] = 0.0
        return z_

    def grad_x(self, z):
        dz1_dx = self._y_layers[0].weight
        z0 = z.clone()
        z = self._y_layers[0](z0)
        dz1r_dz1 = self.relu_grad(z)
        dz1r_dz1_ = dz1r_dz1.unsqueeze(2).repeat(1, 1, dz1_dx.shape[1])
        z = self.nonlinearity(z)
        dz1_dx_ = dz1_dx.expand_as(dz1r_dz1_)
        dz1r_dx = dz1r_dz1_ * dz1_dx_

        layer_num = len(self._z_layers)
        for i in range(layer_num):
            z = self._y_layers[i + 1](z0) + self._z_layers[i](z)
            dz1r_dz = self.relu_grad(z)
            z = self.nonlinearity(z)
            dzz1_dx = torch.matmul(torch.exp(self._z_layers[i].log_weight), dz1r_dx)
            dzy1_dx = self._y_layers[i + 1].weight
            dzy1_dx_plus_dzz1_dx = dzy1_dx + dzz1_dx
            dz1r_dz_ = dz1r_dz.unsqueeze(2).repeat(1, 1, dzy1_dx_plus_dzz1_dx.shape[2])
            dz1r_dx = dz1r_dz_ * dzy1_dx_plus_dzz1_dx

        return dz1r_dx.squeeze(1)



class Damping(nn.Module):
    def __init__(self,
                 input_dim,  # input should be velocity
                 hidden_sizes,
                 w_init_offdiag = nn.init.xavier_uniform_,
                 b_init_offdiag = nn.init.zeros_,
                 w_init_diag=nn.init.xavier_uniform_,
                 # w_init_diag_param=0.1,       # todo
                 b_init_diag=nn.init.zeros_,
                 hidden_nonlinearity=torch.tanh,
                 full_mat=True,
                 damp_min = None
                 ):
        super(Damping, self).__init__()

        N = input_dim
        self.offdiag_output_dim = N*(N-1)//2
        self.diag_output_dim = N
        self.output_dim = self.offdiag_output_dim + self.diag_output_dim
        self.full_mat = full_mat
        if damp_min.shape != input_dim:
            print('damp_min not of correct shape')
            AssertionError
        else:
            self.damp_min = damp_min

        # diag
        self._diag_layers = nn.ModuleList()
        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            linear_layer = nn.Linear(prev_size, size)
            # w_init_diag(linear_layer.weight, w_init_diag_param)
            w_init_diag(linear_layer.weight)
            b_init_diag(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)
            hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))
            self._diag_layers.append(hidden_layers)
            prev_size = size

        self._diag_output_layer = nn.ModuleList()
        linear_layer = nn.Linear(prev_size, self.diag_output_dim)
        # w_init_diag(linear_layer.weight, w_init_diag_param)
        w_init_diag(linear_layer.weight)
        b_init_diag(linear_layer.bias)
        self._diag_output_layer.append(linear_layer)

        # offdiag
        if self.full_mat:
            self._offdiag_layers = nn.ModuleList()
            prev_size = input_dim
            for size in hidden_sizes:
                hidden_layers = nn.Sequential()
                linear_layer = nn.Linear(prev_size, size)
                w_init_offdiag(linear_layer.weight)
                b_init_offdiag(linear_layer.bias)
                hidden_layers.add_module('linear', linear_layer)
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))
                self._offdiag_layers.append(hidden_layers)
                prev_size = size

            self._offdiag_output_layer = nn.ModuleList()
            linear_layer = nn.Linear(prev_size, self.offdiag_output_dim)
            w_init_offdiag(linear_layer.weight)
            b_init_offdiag(linear_layer.bias)
            self._offdiag_output_layer.append(linear_layer)
        print('damper init')

    def forward(self, input):
        # todo this method is not ready for batch input data
        # x = input.view(1,-1)
        x = input
        z = x.clone()
        x0 = x.clone()

        for layer in self._diag_layers:
            x = layer(x)
        x = self._diag_output_layer[0](x)
        # x = torch.exp(x) * x0
        x = (torch.relu(x) + self.damp_min) * x0

        if not self.full_mat:
            assert(x.shape == x0.shape)
            return x
        else:
            n = self.diag_output_dim
            diag_idx = np.diag_indices(n)
            off_diag_idx = np.tril_indices(n, k=-1)
            D = torch.zeros(x0.shape[0], n)

            for layer in self._offdiag_layers:
                z = layer(z)
            z = self._offdiag_output_layer[0](z)

            for i in range(x0.shape[0]):
                L = torch.zeros(n, n)
                diag_elements = x[i]
                off_diag_elements = z[i]
                L[off_diag_idx] = off_diag_elements
                L[diag_idx] = diag_elements
                D_temp = L@L.t()
                D[i] = D_temp @ x0[i]
            return D



