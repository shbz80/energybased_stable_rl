import torch
from torch import nn

class EnergyBasedInitModule(nn.Module):
    """GaussianPSMLPModule with a mean network and only variance parameter for parameter space

    Args:
    """

    def __init__(self,
                 S_param,
                 D_param,
                 std,
                 state_dim,
                 action_dim
                 ):
        super().__init__()

        self.S_param=S_param
        self.D_param=D_param
        self.std=std
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.coord_dim=self.state_dim//2

    def _get_action(self, *inputs):
        state = inputs[0]
        assert(len(state.shape)==2)
        assert(state.shape[1]==self.state_dim)
        x = state[:,:self.coord_dim]
        x_dot = state[:,self.coord_dim:]

        S_param_s = torch.normal(self.S_param, self.std)

        self.u_pot = - S_param_s * x
        self.u_damp = - self.D_param * x_dot
        self.u_quad = torch.zeros(self.coord_dim)

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
