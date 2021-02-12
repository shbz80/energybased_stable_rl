import numpy as np
from garage.torch import NonLinearity

import torch
import torch.nn as nn

# all netwrok settings are done directly here. todo

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())

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


    def init(self, w_init, b_init, w_init_const=None):
        for p in self.p_list:
            if w_init_const:
                w_init(p, w_init_const)
            else:
                w_init(p)
        b_init(self.bias)

    def forward(self, input):
        weights = [p for p in self.p_list]
        if self.col:
            weight = torch.cat(weights, dim=1)
        else:
            weight = torch.cat(weights, dim=0)
        return nn.functional.linear(input, weight, self.bias)

    def get_weight(self):
        weights = [p for p in self.p_list]
        if self.col:
            weight = torch.cat(weights, dim=1)
        else:
            weight = torch.cat(weights, dim=0)
        return weight

    def get_bias(self):
        return self.bias


class PositiveLinearList(LinearList):

    def __init__(self, input_dim, output_dim, col=True):
        super(LinearList, self).__init__()
        self.p_list = nn.ParameterList()
        self.col = col
        self.output_dim = output_dim
        for i in range(input_dim if col else output_dim):
            if col:
                p = nn.Parameter(torch.Tensor(output_dim , 1))
            else:
                p = nn.Parameter(torch.Tensor(1, input_dim))
            self.p_list.append(p)


    def init(self, w_init, w_init_const=None):
        for p in self.p_list:
            if w_init_const:
                w_init(p, w_init_const)
            else:
                w_init(p)

    def forward(self, input):
        weights = [p for p in self.p_list]
        if self.col:
            weight = torch.cat(weights, dim=1)
        else:
            weight = torch.cat(weights, dim=0)
        return nn.functional.linear(input, weight.exp())

    def get_weight(self):
        weights = [p for p in self.p_list]
        if self.col:
            weight = torch.cat(weights, dim=1)
        else:
            weight = torch.cat(weights, dim=0)
        return weight.exp()

# log normal mean and std calculation


desired_lognormal_mean = torch.tensor(0.1)         #todo

class ICNN(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_sizes,
                 w_init=nn.init.xavier_normal_,
                 b_init=nn.init.zeros_,
                 nonlinearity=torch.relu,
                 init_std=0.1
                 ):
        super(ICNN, self).__init__()

        # log normal stuff
        self.desired_mean = desired_lognormal_mean
        self.desired_std = init_std
        self.log_mean = torch.log((self.desired_mean ** 2) / torch.sqrt((self.desired_mean ** 2 + self.desired_std ** 2)))
        self.log_std = torch.sqrt(torch.log(1 + (self.desired_std ** 2) / (self.desired_mean ** 2)))

        self._y_layers = nn.ModuleList()
        for size in hidden_sizes:
            linear_layer = LinearList(input_dim, size, col=True)
            linear_layer.init(nn.init.xavier_uniform_, nn.init.zeros_)
            self._y_layers.append(linear_layer)
        linear_layer = LinearList(input_dim, 1, col=False)
        linear_layer.init(nn.init.xavier_uniform_, nn.init.zeros_)
        self._y_layers.append(linear_layer)

        self._z_layers = nn.ModuleList()
        prev_size = hidden_sizes[0]
        for size in hidden_sizes[1:]:
            positive_linear_layer = PositiveLinearList(prev_size, size, col=True)
            positive_linear_layer.init(nn.init.constant_, self.log_mean)          #todo
            # positive_linear_layer.init(nn.init.xavier_uniform_)
            self._z_layers.append(positive_linear_layer)
            prev_size = size
        positive_linear_layer = PositiveLinearList(prev_size, 1, col=False)
        positive_linear_layer.init(nn.init.constant_, self.log_mean)
        # positive_linear_layer.init(nn.init.xavier_uniform_)
        self._z_layers.append(positive_linear_layer)

        self.nonlinearity = NonLinearity(torch.relu)
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

        dz1_dx = self._y_layers[0].get_weight()
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
            dzz1_dx = torch.matmul(self._z_layers[i].get_weight(), dz1r_dx)
            dzy1_dx = self._y_layers[i + 1].get_weight()
            dzy1_dx_plus_dzz1_dx = dzy1_dx + dzz1_dx
            dz1r_dz_ = dz1r_dz.unsqueeze(2).repeat(1, 1, dzy1_dx_plus_dzz1_dx.shape[2])
            dz1r_dx = dz1r_dz_ * dzy1_dx_plus_dzz1_dx

        return dz1r_dx.squeeze(1)



class Damping(nn.Module):
    def __init__(self,
                 input_dim,  # input should be velocity
                 hidden_sizes,
                 w_init=nn.init.xavier_normal_,
                 w_init_const=1.0,
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

        # diag
        self._diag_layers = nn.ModuleList()
        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            linear_layer = LinearList(prev_size, size, col=True)
            linear_layer.init(nn.init.xavier_uniform_, nn.init.zeros_)
            hidden_layers.add_module('linear', linear_layer)
            hidden_layers.add_module('non_linearity',
                                         NonLinearity(torch.tanh))
            self._diag_layers.append(hidden_layers)
            prev_size = size

        self._diag_output_layer = nn.ModuleList()
        linear_layer = LinearList(prev_size, self.diag_output_dim, col=False)
        # linear_layer.init(nn.init.constant_, nn.init.zeros_, log_const_init)      # todo
        linear_layer.init(nn.init.xavier_uniform_, nn.init.zeros_)
        self._diag_output_layer.append(linear_layer)

        # offdiag
        if self.full_mat:
            self._offdiag_layers = nn.ModuleList()
            prev_size = input_dim
            for size in hidden_sizes:
                hidden_layers = nn.Sequential()
                linear_layer = LinearList(prev_size, size, col=True)
                linear_layer.init(nn.init.xavier_uniform_, nn.init.zeros_)
                hidden_layers.add_module('linear', linear_layer)
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(torch.tanh))
                self._offdiag_layers.append(hidden_layers)
                prev_size = size

            self._offdiag_output_layer = nn.ModuleList()
            linear_layer = LinearList(prev_size, self.offdiag_output_dim, col=False)
            linear_layer.init(nn.init.xavier_uniform_, nn.init.zeros_)
            self._offdiag_output_layer.append(linear_layer)
        print('damper init')

    def forward(self, input):
        # todo this method is not ready for batch input data
        # x = input.view(1,-1)
        x = input
        z = input
        x0 = x.clone()

        for layer in self._diag_layers:
            x = layer(x)
        x = self._diag_output_layer[0](x)

        if not self.full_mat:
            assert(x.shape == x0.shape)
            return torch.exp(x) * x0
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

# class Damping(nn.Module):
#     def __init__(self,
#                  input_dim,  # input should be velocity
#                  hidden_sizes,
#                  w_init=nn.init.xavier_normal_,
#                  w_init_const=1.0,
#                  b_init=nn.init.zeros_,
#                  hidden_nonlinearity=torch.tanh,
#                  full_mat=True,
#                  ):
#         super(Damping, self).__init__()
#
#         N = input_dim
#         self.offdiag_output_dim = N*(N-1)//2
#         self.diag_output_dim = N
#         self.output_dim = self.offdiag_output_dim + self.diag_output_dim
#         self.full_mat = full_mat
#
#         self._layers = nn.ModuleList()
#
#         prev_size = input_dim
#         for size in hidden_sizes:
#             hidden_layers = nn.Sequential()
#             linear_layer = LinearList(prev_size, size, col=True)
#             linear_layer.init(w_init, b_init, w_init_const)
#             hidden_layers.add_module('linear', linear_layer)
#             hidden_layers.add_module('non_linearity',
#                                          NonLinearity(hidden_nonlinearity))
#             self._layers.append(hidden_layers)
#             prev_size = size
#
#         if self.full_mat:
#             out_dim = self.output_dim
#         else:
#             out_dim = self.diag_output_dim
#
#         self._output_layer = nn.ModuleList()
#         linear_layer = LinearList(prev_size, out_dim, col=False)
#         linear_layer.init(w_init, b_init, w_init_const)
#         self._output_layer.append(linear_layer)
#
#     def forward(self, input):
#         # todo this method is not ready for batch input data
#         # x = input.view(1,-1)
#         x = input
#         x0 = x.clone()
#         for layer in self._layers:
#             x = layer(x)
#
#         x = self._output_layer[0](x)
#         n = self.diag_output_dim
#         diag_idx = np.diag_indices(n)
#         D = torch.zeros(x0.shape[0], n)
#
#         for i in range(x.shape[0]):
#             L = torch.zeros(n, n)
#             if self.full_mat:
#                 diag_elements = torch.exp(x[i, :self.diag_output_dim])
#                 off_diag_elements = x[i, self.diag_output_dim:]
#                 off_diag_idx = np.tril_indices(n, k=-1)
#                 L[off_diag_idx] = off_diag_elements
#                 L[diag_idx] = diag_elements
#                 D_temp = L@L.t()
#                 D[i] = D_temp @ x0[i]
#             else:
#                 diag_elements = torch.exp(x[i])
#                 D[i] = torch.diag(diag_elements) @ x0[i]
#
#         # D = torch.exp(x) * x0
#         return D


