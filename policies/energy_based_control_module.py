import torch
from torch import nn
from torch.distributions import Normal, MultivariateNormal
from garage.torch.distributions import TanhNormal
from torch.distributions.independent import Independent
from energybased_stable_rl.utilities.diff import jacobian # the jacobian API in pyTorch is not used because it requires a
                                                    # function to be passed
from energybased_stable_rl.policies.energy_control_modules import ICNN, Damping

class GaussianEnergyBasedModule(nn.Module):
    """GaussianPSMLPModule with a mean network and only variance parameter for parameter space

    Args:
    """

    def __init__(self,
                 coord_dim,
                 icnn_hidden_sizes=(32, 32),
                 damper_hidden_sizes=(32, 32),
                 w_init=nn.init.xavier_uniform_,
                 b_init=nn.init.zeros_,
                 icnn_hidden_nonlinearity=torch.relu,       # this has to a convex function e.g. relu
                 damper_hidden_nonlinearity=torch.tanh,
                 damper_full_mat = True,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 full_std=False,
                 jac_update_rate = 10,
                 std_parameterization='exp',
                 init_quad_pot=1.0,
                 min_quad_pot=1e-3,
                 max_quad_pot=1e1,
                 icnn_min_lr = 1e-1
                 ):
        super().__init__()
        self._coord_dim = coord_dim
        self._state_dim = coord_dim*2
        self._action_dim = coord_dim
        self._icnn_hidden_sizes = icnn_hidden_sizes
        self._damper_hidden_sizes = damper_hidden_sizes
        self._w_init = w_init
        self._b_init = b_init
        self._icnn_hidden_nonlinearity = icnn_hidden_nonlinearity
        self._damper_hidden_nonlinearity = damper_hidden_nonlinearity
        self._damper_full_mat = damper_full_mat
        self._learn_std = learn_std
        self._min_std = min_std
        self._max_std = max_std
        self._full_std = full_std
        if full_std == True:
            self._normal_distribution_cls = MultivariateNormal
        else:
            self._normal_distribution_cls = Normal
        self._jac_update_rate = jac_update_rate
        self._std_parameterization = std_parameterization

        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        init_std_param = torch.Tensor([init_std]).log()
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param
            self.register_buffer('init_std', self._init_std)

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
            self.register_buffer('min_std_param', self._min_std_param)
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()
            self.register_buffer('max_std_param', self._max_std_param)

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
            w_init=w_init,
            b_init=b_init,
            nonlinearity=icnn_hidden_nonlinearity
        )

        self._damping_module = Damping(
            coord_dim,
            hidden_sizes = damper_hidden_sizes,
            w_init=w_init,
            b_init=b_init,
            hidden_nonlinearity=damper_hidden_nonlinearity,
            full_mat=damper_full_mat
        )

        self.curr_x_min = torch.rand(self._coord_dim)
        self.curr_f_min = 0.
        self._icnn_min_lr = icnn_min_lr

        # self._min_icnn()


    def _min_icnn(self):
        with torch.enable_grad():
            x = self.curr_x_min
            x.requires_grad = True
            max_iter = 100
            optimizer = torch.optim.LBFGS([x], lr=self._icnn_min_lr, max_iter=max_iter, line_search_fn='strong_wolfe')

            def closure():
                optimizer.zero_grad()
                loss = self._icnn_module(x)
                loss.backward()
                return loss

            optimizer.step(closure)
            print('ICNN min done')
            state_dict = optimizer.state_dict()
            n_iter = state_dict['state'][0]['n_iter']
            assert(n_iter <  max_iter)
        self.curr_x_min = x.detach()
        self.curr_f_min = self._icnn_module(self.curr_x_min)
        del optimizer

    def _get_action(self, *inputs):

        state = inputs[0]
        assert(len(state.shape)==2)
        assert(state.shape[1]==self._state_dim)
        x = state[:,:self._coord_dim] - self.curr_x_min
        x_dot = state[:,self._coord_dim:]
        u_pot = torch.zeros(x.shape)
        u_damp = torch.zeros(x_dot.shape)

        # should the param be clammped and stored? todo
        quad_pot = self._quad_pot_param.clamp(
            min=(self._min_quad_pot_param.item()),
            max=(self._max_quad_pot_param.item()))
        quad_pot = quad_pot.exp()

        with torch.enable_grad():
            for n in range(x.shape[0]):
                x_n = x[n]
                x_n.requires_grad = True
                psi_n = self._icnn_module(x_n)
                u_pot[n] = -jacobian(psi_n, x_n, create_graph=True) - torch.diag(quad_pot) @ x_n
                u_damp[n] = -self._damping_module(x_dot[n]) @ x_dot[n]
            u = u_pot + u_damp
            # u = u_pot

            return u

    def _get_mean_and_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        assert len(inputs) == 1

        if self._min_std_param or self._max_std_param:
            std = self._init_std.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = std.exp()
        else:
            std = std.exp().exp().add(1.).log()

        with torch.enable_grad():
            mean = self._get_action(*inputs)
            param_list = []
            for param in self.named_parameters():
                if '_init_std' in param:
                    pass
                else:
                    param_list.append(param[1])

            if self._full_std == True:
                var_shape = list(mean.shape) + [self._action_dim]
                var = torch.zeros(var_shape)
            else:
                var_shape = mean.shape
                var = torch.zeros(var_shape)
            for n in range(mean.shape[0]):
                if not(n % self._jac_update_rate):
                    Jl = []
                    for param in param_list:
                        Jn = jacobian(mean[n], param, create_graph=False)
                        Jl.append(Jn.view(Jn.shape[0],-1))

                    J = torch.cat(Jl, dim=1).detach()

                if self._full_std == True:
                    Sigma = torch.eye(J.shape[1]) * std ** 2
                    var[n] = J @ Sigma @ J.t()
                else:
                    Sigma = torch.eye(J.shape[1]) * std
                    var[n] = torch.diag(J @ Sigma @ J.t())

        return mean, var



    def forward(self, *inputs):
        """Forward method.

                Args:
                    *inputs: Input to the module.

                Returns:
                    torch.distributions.independent.Independent: Independent
                        distribution.

                """
        mean, var = self._get_mean_and_std(*inputs)
        dist = self._normal_distribution_cls(mean, var)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal) and isinstance(dist, Normal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist
