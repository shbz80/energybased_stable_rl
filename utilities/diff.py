import torch
from energybased_stable_rl.policies.energy_control_modules import ICNN
# from energybased_stable_rl.policies.energy_based_control_module import EnergyBasedControlModule
from torch import nn
import traceback

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


# a = torch.rand((4,2), requires_grad=False)
# x = torch.rand(2, requires_grad=True)
#
# def f(x):
#     return a @ x
#
# y = f(x)
#
# J = jacobian(y, x)
# None

def smooth_relu(x):
    d = torch.tensor(.01)
    z = torch.zeros_like(x)
    if torch.any(x <= 0.0):
        z[(x <= 0.0)] = 0.0

    if torch.any(torch.logical_and(x > 0.0, x < d)):
        z[torch.logical_and(x > 0.0, x < d)] = x[torch.logical_and(x > 0.0, x < d)]**2/(2.0*d)

    if torch.any(x >= d):
        z[(x >= d)] = x[(x >= d)]-(d/2.0)

    return z

icnn_module = ICNN(
            2,
            hidden_sizes = (8,8),
            w_init_y=nn.init.xavier_uniform_,
            b_init_y=nn.init.zeros_,
            w_init_z=nn.init.constant_,
            w_init_z_param=0.1,
            nonlinearity=smooth_relu
        )

x = torch.rand((2,2), requires_grad=True)
print('x',x)
# x.requires_grad = True
psi = icnn_module(x)
print('psi',psi)
start_time = time.time()
J = jacobian_batch(psi, x, create_graph=True)
j_time = time.time()-start_time
# print('J time', j_time)
# start_time = time.time()
# for i in range(x.shape[0]):
#     grad = icnn_module.grad_x(x[i].view(1,-1))
#     print('grad', grad)
# print('grad time', time.time()-start_time)
print('J',J)
start_time = time.time()
x_grad = icnn_module.grad_x(x)
grad_time = time.time()-start_time
print('x_grad', x_grad)
# print('grad time', grad_time)
# print('comp time ratio', j_time/grad_time)
None

# icnn_module = ICNN(
#             2,
#             hidden_sizes = (8,8),
#             w_init_y=nn.init.xavier_uniform_,
#             b_init_y=nn.init.normal_,
#             w_init_z=nn.init.constant_,
#             w_init_z_param=0.1,
#             nonlinearity=torch.relu
#         )
#
# print('ICNN min begin')
# with torch.enable_grad():
#     x = torch.tensor([0.9765, 0.9248])
#     print('x',x)
#     x.requires_grad = True
#     max_iter = 1000
#     optimizer = torch.optim.LBFGS([x], lr=1e-1, max_iter=max_iter, line_search_fn='strong_wolfe')  # todo remove line search if error persists
#
#     def closure():
#         optimizer.zero_grad()
#         loss = icnn_module(x)
#         loss.backward()
#         return loss
#     try:
#         optimizer.step(closure)
#     except Exception:
#         traceback.print_exc()
#
#     state_dict = optimizer.state_dict()
#     n_iter = state_dict['state'][0]['n_iter']
#     print('icnn n_iter',n_iter)
#     if (n_iter >=  max_iter):
#         print('ICNN min not done')
#         assert(False)
#     else:
#         print('ICNN min done')
# curr_x_min = x.detach()
# print(curr_x_min, icnn_module(curr_x_min))


# module = EnergyBasedControlModule(
#             coord_dim = 2,
#             icnn_hidden_sizes=(16, 16),
#             w_init_icnn_y=nn.init.xavier_uniform_,
#             b_init_icnn_y=nn.init.normal_,
#             w_init_icnn_z=nn.init.constant_,
#             w_init_icnn_z_param=0.1,
#             nonlinearity_icnn=torch.relu,
#             damper_hidden_sizes=(8, 8),
#             w_init_damper_offdiag=nn.init.xavier_uniform_,
#             b_init_damper_offdiag=nn.init.zeros_,
#             w_init_damper_diag=nn.init.constant_,
#             w_init_damper_diag_param=0.1,
#             b_init_damper_diag=nn.init.zeros_,
#             hidden_nonlinearity_damper=torch.tanh,
#             full_mat_damper_=True,
#             init_quad_pot=1.0,
#             min_quad_pot=1e-3,
#             max_quad_pot=1e1,
#             icnn_min_lr=1e-1,
#         )
# module._min_icnn()
# print('x_min',module.curr_x_min, module.curr_f_min)