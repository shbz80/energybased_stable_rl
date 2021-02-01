import torch
from torch import nn
from torch.distributions import Normal, MultivariateNormal
from garage.torch.distributions import TanhNormal
from torch.distributions.independent import Independent
from energybased_stable_rl.utilities.diff import jacobian # the jacobian API in pyTorch is not used because it requires a
                                                    # function to be passed
from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules import GaussianMLPBaseModule

class GaussianPSMLPModule(GaussianMLPBaseModule):
    """GaussianPSMLPModule with a mean network and only variance parameter for parameter space

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for parameter space std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 full_std=False,
                 jac_update_rate = 10,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        self.full_std = full_std
        if full_std == True:
            self.normal_distribution_cls = MultivariateNormal
        else:
            self.normal_distribution_cls = Normal
        self.jac_update_rate = jac_update_rate
        super(GaussianPSMLPModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=self.normal_distribution_cls)

        self._mean_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization)



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
            mean = self._mean_module(*inputs)
            if self.full_std == True:
                var_shape = list(mean.shape) + [self._action_dim]
                var = torch.zeros(var_shape)
            else:
                var_shape = mean.shape
                var = torch.zeros(var_shape)
            for n in range(mean.shape[0]):
                if not(n % self.jac_update_rate):
                    Jl = []
                    for param in self._mean_module.parameters():
                        Jn = jacobian(mean[n], param, create_graph=True)
                        Jl.append(Jn.view(Jn.shape[0],-1))

                    J = torch.cat(Jl, dim=1)

                if self.full_std == True:
                    Sigma = torch.eye(J.shape[1]) * std ** 2
                    var[n] = J @ Sigma @ J.t()
                else:
                    Sigma = torch.eye(J.shape[1]) * std
                    var[n] = torch.diag(J @ Sigma @ J.t())

        return mean, var

    def _get_action(self, *inputs):
        return self._mean_module(*inputs)

    def forward(self, *inputs):
        """Forward method.

                Args:
                    *inputs: Input to the module.

                Returns:
                    torch.distributions.independent.Independent: Independent
                        distribution.

                """
        mean, var = self._get_mean_and_std(*inputs)
        dist = self.normal_distribution_cls(mean, var)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal) and isinstance(dist, Normal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist
