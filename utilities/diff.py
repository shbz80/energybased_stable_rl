import torch
from energybased_stable_rl.policies.energy_control_modules import ICNN
from torch import nn
import time
def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)

def jacobian_batch(y, x, create_graph=False):
    assert(y.shape[0]==x.shape[0])
    assert(y.shape[1]==1)
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape)[i])
        grad_y[i] = 0.
    J = torch.stack(jac)
    assert(J.shape==x.shape)
    return J


# x = torch.rand((4,4), requires_grad=True)
# a = torch.rand(4, requires_grad=False)
#
# def f(x):
#     return x @ a
#
# y = f(x)
#
# J = jacobian(y, x)
# None


# icnn_module = ICNN(
#             2,
#             hidden_sizes = (8,8),
#             w_init=nn.init.xavier_uniform_,
#             b_init=nn.init.zeros_,
#             nonlinearity=torch.relu
#         )
#
# x = torch.rand(1000,2)
# print('x',x)
# x.requires_grad = True
# psi = icnn_module(x)
# print('psi',psi)
# start_time = time.time()
# J = jacobian_batch(psi, x, create_graph=True)
# j_time = time.time()-start_time
# print('J time', j_time)
# # start_time = time.time()
# # for i in range(x.shape[0]):
# #     grad = icnn_module.grad_x(x[i].view(1,-1))
# #     print('grad', grad)
# # print('grad time', time.time()-start_time)
# # print('J',J)
# start_time = time.time()
# x_grad = icnn_module.grad_x(x)
# grad_time = time.time()-start_time
# # print('x_grad', x_grad)
# print('grad time', grad_time)
# print('comp time ratio', j_time/grad_time)
# None