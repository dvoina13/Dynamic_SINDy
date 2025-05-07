import torch
import torch.nn as nn
from torch.autograd import Variable
import math

from args_params_hyperparams import parse_hyperparams, parse_args
args = parse_args()
hyperparams = parse_hyperparams()

def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))

class _L0Norm(nn.Module):

    def __init__(self, library_dim, z_dim,
                 loc_mean=0, loc_sdev=0.01,
                 beta=2 / 3, gamma=-0.1,
                 zeta=1.1, fix_temp=True):
        """
        Base class of layers using L0 Norm
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()
        self.library_dim = library_dim
        self.z_dim = z_dim
        self.loc = nn.Parameter(torch.zeros([self.library_dim, self.z_dim]).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros([self.library_dim, self.z_dim]))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)
        self.sig = nn.Sigmoid()

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = self.sig((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = self.sig(self.loc - self.temp * self.gamma_zeta_ratio).sum()#.sum(1).mean()
        else:
            s = self.sig(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return hard_sigmoid(s), penalty


def smoothness_reg(sindy_coeffs):
    norm = sindy_coeffs.size(0) * sindy_coeffs.size(1) * sindy_coeffs.size(2) * sindy_coeffs.size(3)
    sindy_coeffs_dt = ((sindy_coeffs[:, 1:, :, :] - sindy_coeffs[:, :-1, :, :]) ** 2).sum()

    return sindy_coeffs_dt