import torch
from torch import nn
from energybased_stable_rl.utilities.diff import jacobian_batch # the jacobian API in pyTorch is not used because it requires a todo
                                                    # function to be passed
from energybased_stable_rl.policies.energy_control_modules import ICNN, Damping
import time
import traceback


class EnergyBasedControlModule(nn.Module):
    """GaussianPSMLPModule with a mean network and only variance parameter for parameter space

    Args:
    """

    def __init__(self,
                 coord_dim,
                 icnn_hidden_sizes=(32, 32),
                 w_init_icnn_y=nn.init.xavier_uniform_,
                 b_init_icnn_y=nn.init.zeros_,
                 w_init_icnn_z=nn.init.constant_,
                 w_init_icnn_z_param=0.1,
                 nonlinearity_icnn=torch.relu,
                 damper_hidden_sizes=(32, 32),
                 w_init_damper_offdiag=nn.init.xavier_uniform_,
                 b_init_damper_offdiag=nn.init.zeros_,
                 w_init_damper_diag=nn.init.xavier_uniform_,
                 w_init_damper_diag_param=0.1,
                 b_init_damper_diag=nn.init.zeros_,
                 hidden_nonlinearity_damper=torch.tanh,
                 full_mat_damper=True,
                 damp_min = None,
                 init_quad_pot=1.0,
                 min_quad_pot=1e-3,
                 max_quad_pot=1e1,
                 icnn_min_lr = 1e-1,
                 ):
        super().__init__()
        self._coord_dim = coord_dim
        self._state_dim = coord_dim*2
        self._action_dim = coord_dim
        self.init_quad_pot = 1.0,
        self.min_quad_pot = 1e-3,
        self.max_quad_pot = 1e1,
        self.icnn_min_lr = 1e-1,

        assert(min_quad_pot>0 and max_quad_pot>0)
        assert(init_quad_pot>min_quad_pot and max_quad_pot>init_quad_pot)
        init_quad_pot_param = torch.ones(coord_dim)*torch.Tensor([init_quad_pot]).log()
        self._quad_pot_param = torch.nn.Parameter(init_quad_pot_param)

        self._min_quad_pot_param = self._max_quad_pot_param = None
        if min_quad_pot is not None:
            self._min_quad_pot_param = torch.Tensor([min_quad_pot]).log()
            self.register_buffer('min_quad_pot_param', self._min_quad_pot_param)
        if max_quad_pot is not None:
            self._max_quad_pot_param = torch.Tensor([max_quad_pot]).log()
            self.register_buffer('max_quad_pot_param', self._max_quad_pot_param)

        self._icnn_module = ICNN(
            coord_dim,
            hidden_sizes = icnn_hidden_sizes,
            w_init_y=w_init_icnn_y,
            b_init_y=b_init_icnn_y,
            w_init_z=w_init_icnn_z,
            w_init_z_param=w_init_icnn_z_param,
            nonlinearity=nonlinearity_icnn,
        )

        self._damping_module = Damping(
            coord_dim,
            hidden_sizes = damper_hidden_sizes,
            w_init_offdiag=w_init_damper_offdiag,
            b_init_offdiag=b_init_damper_offdiag,
            w_init_diag=w_init_damper_diag,
            # w_init_diag_param=w_init_damper_diag_param,       #todo
            b_init_diag=b_init_damper_diag,
            hidden_nonlinearity=hidden_nonlinearity_damper,
            full_mat=full_mat_damper,
            damp_min=damp_min
        )

        self.curr_x_min = torch.rand(self._coord_dim)
        self.curr_f_min = 0.
        self._icnn_min_lr = icnn_min_lr
        self.u_pot = None
        self.u_quad = None
        self.u_damp = None

    def _min_icnn(self):
        print('ICNN min begin')
        with torch.enable_grad():
            x = self.curr_x_min
            x.requires_grad = True
            max_iter = 1000
            optimizer = torch.optim.LBFGS([x], lr=self._icnn_min_lr, max_iter=max_iter, line_search_fn='strong_wolfe')  # todo remove line search if error persists

            def closure():
                optimizer.zero_grad()
                loss = self._icnn_module(x)
                loss.backward()
                return loss
            try:
                optimizer.step(closure)
            except Exception:
                traceback.print_exc()

            state_dict = optimizer.state_dict()
            n_iter = state_dict['state'][0]['n_iter']
            print('icnn n_iter',n_iter)
            if (n_iter >=  max_iter):
                print('ICNN min not done')
                assert(False)
            # else:
            #     print('ICNN min done')
        x_min = x.detach()
        f_min = self._icnn_module(x_min)
        del optimizer
        return x_min, f_min

    def min_icnn(self):
        f_min_l = []
        x_min_l = []
        for i in range(5):
            x_min, f_min = self._min_icnn()
            f_min_l.append(f_min)
            x_min_l.append(x_min)

        arg_min = torch.argmin(torch.cat(f_min_l))

        self.curr_x_min = x_min_l[arg_min]
        self.curr_f_min = f_min_l[arg_min]

        print('f_min_l',f_min_l)


    def _get_action(self, *inputs):
        state = inputs[0]
        assert(len(state.shape)==2)
        assert(state.shape[1]==self._state_dim)
        x = state[:,:self._coord_dim] - self.curr_x_min
        x_dot = state[:,self._coord_dim:]


        # should the param be clammped and stored? todo
        quad_pot = self._quad_pot_param.clamp(
            min=(self._min_quad_pot_param.item()),
            max=(self._max_quad_pot_param.item()))
        if torch.any(torch.isnan(quad_pot)):
            print('nan in quad_pot')
            assert(False)
        quad_pot = quad_pot.exp()


        with torch.enable_grad():
            x.requires_grad = True
            psi = self._icnn_module(x)
            self.u_pot = - jacobian_batch(psi, x, create_graph=False).detach()
        # self.u_pot = -self._icnn_module.grad_x(x)
        if torch.any(torch.isnan(self.u_pot)):
            print('nan in u_pot')
            assert (False)
        if torch.any((self.u_pot>10.0)):
            print('Warning: large u_pot', self.u_pot)

        self.u_quad = -torch.diag(quad_pot) @ x.t()
        self.u_quad.t_()
        if torch.any(self.u_quad > 5.0):
            print('Warning: large u_quad', self.u_quad)

        self.u_damp = -self._damping_module(x_dot)   # todo
        # self.u_damp = -x_dot*2.0
        if torch.any(torch.isnan(self.u_damp)):
            print('nan in u_damp')
            assert (False)
        if torch.any((self.u_damp>5.0)):
            print('Warning: large u_damp', self.u_damp)

        return self.u_pot, self.u_quad, self.u_damp

    def forward(self, *inputs):
        """Forward method.

                Args:
                    *inputs: Input to the module.

                Returns:
                    torch.distributions.independent.Independent: Independent
                        distribution.

                """
        return None
