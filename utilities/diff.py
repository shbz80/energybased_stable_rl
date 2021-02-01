import torch


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


a = torch.rand((4,4), requires_grad=True)
b = torch.rand(4, requires_grad=False)

def f(b):
    return a @ b

y = f(b)
J = jacobian(y, a)
# J = jacobian(y, a.view(-1))
None

