import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from regularization import _L0Norm
from utils1 import library_size, prod_dim, get_weight_dim
from utils1 import sindy_library

from args_params_hyperparams import parse_hyperparams, parse_args
args = parse_args()
hyperparams = parse_hyperparams()

class Encoder(nn.Module):
    def __init__(self, hyperparams, args, out_dim, hidden_dim=64, bias=True, activation=nn.ELU(),
                 norm='batch'):
        super(Encoder, self).__init__()

        self.in_dim = 2*args.z_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        encoder_layers = []
        encoder_layers.append(nn.Linear(self.in_dim, self.hidden_dim))
        encoder_layers.append(activation)
        encoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        encoder_layers.append(activation)
        encoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        encoder_layers.append(activation)
        encoder_layers.append(nn.Linear(self.hidden_dim, self.out_dim))

        self.encoder_layers = nn.Sequential(*encoder_layers)

        #self.fc1 = nn.Linear(self.out_dim, int(self.out_dim//2))
        #self.fc2 = nn.Linear(self.out_dim, int(self.out_dim//2))

    def forward(self, x, x_dot):

        x = x.view(-1, x.size()[-1]); x_dot = x_dot.view(-1, x_dot.size()[-1])
        print(x.size(), x_dot.size())
        x_vec = torch.cat((x, x_dot), dim=1)
        out = self.encoder_layers(x_vec)

        print("out size", out.size())
        return out[:, :int(self.out_dim//2)], out[:, int(self.out_dim//2):]
        #return self.fc1(out), self.fc2(out)
class HyperNet(nn.Module):
    def __init__(self, in_dim, out_shape, random=True, hidden_dims=[8, 16, 32], bias=True, activation=nn.ELU(),
                 norm='batch'):
        super(HyperNet, self).__init__()

        self.in_dim = in_dim
        self.out_shape = out_shape
        self.number_of_layers = len(hidden_dims)
        self.random = random

        in_features = self.in_dim
        layers = []; layers2 = [];
        layer_numbers = [];
        count = 0
        for out_features in hidden_dims:
            layers.append(nn.Linear(in_features, out_features, bias=True))
            layer_numbers.append(count);
            count += 1
            # if norm == 'batch':
            #    layers.append(nn.BatchNorm1d(out_features))
            # elif norm == 'layer':
            #    layers.append(nn.LayerNorm(out_features))
            # layer_numbers.append(count); count+=1
            layers.append(activation)
            count += 1;

            in_features = out_features

        layers.append(nn.Linear(in_features, np.prod(self.out_shape), bias=bias))
        layer_numbers.append(count)
        self.layer_numbers = layer_numbers

        self.layers = nn.Sequential(*layers)

        """
        in_features = self.in_dim
        count = 0
        for out_features in hidden_dims:
            layers2.append(nn.Linear(in_features, out_features, bias=True))
            count += 1
            layers2.append(activation)
            count += 1;

            in_features = out_features

        layers2.append(nn.Linear(in_features, np.prod(self.out_shape), bias=bias))
        self.layers2 = nn.Sequential(*layers2)

        self.hypernet_mask = nn.Parameter(torch.ones(self.out_shape), requires_grad = False)
        """
    def forward_hypernet(self, Weights, n=None, batch_size=1, hypernets =1, device="cpu"):
        if n is None:
            if self.random:
                print("network is random, OK")
                n = torch.randn((batch_size, self.in_dim), device=device)
            else:
                n = torch.ones((batch_size, self.in_dim)) / self.in_dim
                n = n.to(device)

        x = n

        for i in range(len(Weights)):
            x = x.unsqueeze(2)

            w_dim1, w_dim2 = Weights[i].size(1), Weights[i].size(2)
            weight = self.layers[self.layer_numbers[i]].weight.unsqueeze(0).repeat(batch_size, 1, 1) + Weights[i]
            bias = self.layers[self.layer_numbers[i]].bias.unsqueeze(0).repeat(batch_size, 1)

            x = torch.bmm(weight.to(device), x.to(device)).squeeze()
            if len(x.size()) < 2:
                x = x.unsqueeze(0)
            if bias != None:
                x += bias.to(device)

            #batch norm
            #n_chans = x.shape[0]
            #m = torch.nn.BatchNorm1d(n_chans)
            #x = m(x.permute(1,0)).permute(1,0)

            #if i >= 2:
            #    x = F.dropout(x, p=0.2)

            if i < len(Weights) - 1:
                x = F.elu(x, inplace=False)

        """
        if hypernets == 2:
            xx = self.layers2(n.to(device))
            self.hypernet_mask = nn.Parameter(torch.zeros_like(self.hypernet_mask), requires_grad=False)
            self.hypernet_mask[0,0] = 1; self.hypernet_mask[0,1] = 1;
            print("x xx sizes", x.size(), xx.size())
            x = x * self.hypernet_mask.repeat(batch_size, 1, 1).view(batch_size, -1) + xx * (1 - self.hypernet_mask.repeat(batch_size, 1, 1).view(batch_size, -1))
        """
        return x.reshape(n.size(0), *self.out_shape)

class rHyperNet(nn.Module):
    def __init__(self, shapes, basis, out_shape, num_layers, len_seq, batch_size, mix_dim, x_in_random=False, bias=True,
                 activation=nn.ELU(), norm='batch', lstm_model=True, device=0):
        super(rHyperNet, self).__init__()

        self.lstm_model = lstm_model

        self.shapes = shapes
        self.num_layers = num_layers
        self.len_seq = len_seq
        self.mix_dim = mix_dim
        self.x_in_random = x_in_random

        self.basis = basis
        self.out_shape = out_shape
        self.in_dim = out_shape

        if self.lstm_model:
            self.rnn = nn.LSTM(self.in_dim, self.out_shape, self.num_layers, batch_first=True, bias=False)
        else:
            self.rnn = nn.RNN(self.in_dim, self.out_shape, self.num_layers, nonlinearity='tanh', batch_first=True,
                              bias=False).to(device)

    def forward(self, len_seq, batch_size, z0=None, c0=None, network_input=None, device='cpu'):
        if network_input == None:
            if self.x_in_random:
                print("RNN is random, OK")
                network_input = torch.randn(batch_size, len_seq, self.in_dim)
            else:
                network_input = torch.ones(batch_size, len_seq, self.in_dim) / self.in_dim

        if z0 == None:
            z0 = torch.ones(self.num_layers, batch_size, self.in_dim) / self.in_dim
            c0 = torch.ones(self.num_layers, batch_size, self.in_dim) / self.in_dim

        if self.lstm_model:
            weights, (zn, cn) = self.rnn(network_input.to(device), (z0.to(device), c0.to(device)))
        else:
            weights, zn = self.rnn(network_input.to(device), z0.to(device))
            cn = None;

        Weights = []
        for i in range(len(self.basis)):
            w = weights[:, :, i * self.mix_dim:(i + 1) * self.mix_dim]
            w_dim1, w_dim2 = self.basis[i].size(1), self.basis[i].size(2)
            Weights.append(
                torch.matmul(w.view(-1, self.mix_dim), self.basis[i].view(self.mix_dim, -1)).view(batch_size * len_seq,
                                                                                                  w_dim1, w_dim2))

        return zn, cn, Weights

class Net(nn.Module):
    def __init__(self, args, hyperparams):
        super(Net, self).__init__()

        self.z_dim = args.z_dim
        self.poly_order = args.poly_order

        self.include_constant = args.include_constant
        self.include_sine = args.include_sine
        self.lstm_model = hyperparams.lstm
        self.statistic_batch_size = args.statistic_batch_size
        self.batch_size = hyperparams.batch_size
        self.len_seq = hyperparams.len_seq  # hyperparams["len_seq"]

        self.noise_dim1 = hyperparams.noise_dim1
        self.noise_dim2 = hyperparams.noise_dim2
        self.hypernet_hidden_dim2 = hyperparams.hidden_dim2
        self.num_layers_rnn = hyperparams.num_layers_rnn
        #self.hypernets = hyperparams.hypernets

        self.random = hyperparams.random

        self.library_dim = library_size(self.z_dim, self.poly_order,
                                        include_constant=self.include_constant, use_sine=self.include_sine)

        self.mix_dim = args.mix_dim

        self.autoencoder = hyperparams.autoencoder
        if self.autoencoder:
            self.encoder = Encoder(hyperparams, args, 2 * hyperparams.noise_dim2)

        self.hypernet2 = HyperNet(self.noise_dim2, (self.library_dim, self.z_dim),
                                  self.random, [self.hypernet_hidden_dim2 for _ in range(4)])
        out_shape, all_sizes = prod_dim(get_weight_dim(self.hypernet2)[0])
        self.shapes, self.shapes_no_bias = get_weight_dim(self.hypernet2)
        self.out_shape = out_shape

        self.basis = []
        out_shape_rnn = 0
        for i in range(len(self.shapes)):
            # D_basis = nn.Parameter(torch.randn(self.mix_dim, self.shapes_no_bias[i][0], self.shapes_no_bias[i][1], requires_grad=True)).to(device)
            # nn.init.xavier_uniform_(D_basis).to(device)
            # self.basis.append(D_basis)
            out_shape_rnn += self.mix_dim

        self.D_basis1 = nn.Parameter(
            torch.randn(self.mix_dim, self.shapes_no_bias[0][0], self.shapes_no_bias[0][1], requires_grad=True))
        nn.init.xavier_uniform_(self.D_basis1)
        self.basis.append(self.D_basis1)
        self.D_basis2 = nn.Parameter(
            torch.randn(self.mix_dim, self.shapes_no_bias[1][0], self.shapes_no_bias[1][1], requires_grad=True))
        nn.init.xavier_uniform_(self.D_basis2)
        self.basis.append(self.D_basis2)
        self.D_basis3 = nn.Parameter(
            torch.randn(self.mix_dim, self.shapes_no_bias[2][0], self.shapes_no_bias[2][1], requires_grad=True))
        nn.init.xavier_uniform_(self.D_basis3)
        self.basis.append(self.D_basis3)
        self.D_basis4 = nn.Parameter(
            torch.randn(self.mix_dim, self.shapes_no_bias[3][0], self.shapes_no_bias[3][1], requires_grad=True))
        nn.init.xavier_uniform_(self.D_basis4)
        self.basis.append(self.D_basis4)
        self.D_basis5 = nn.Parameter(
            torch.randn(self.mix_dim, self.shapes_no_bias[4][0], self.shapes_no_bias[4][1], requires_grad=True))
        nn.init.xavier_uniform_(self.D_basis5)
        self.basis.append(self.D_basis5)

        # self.z0_hypernet1 = torch.stack([torch.ones(self.num_layers_rnn, size) for size in all_sizes])
        self.hypernet1 = rHyperNet(shapes=self.shapes_no_bias, basis=self.basis, out_shape=out_shape_rnn,
                                   num_layers=self.num_layers_rnn,
                                   len_seq=self.len_seq, mix_dim=self.mix_dim, x_in_random=self.random,
                                   batch_size=self.batch_size, lstm_model=self.lstm_model)

        self.threshold_mask = nn.Parameter(torch.ones(self.library_dim, self.z_dim),
                                           requires_grad=False)

        self.l0 = _L0Norm(self.library_dim, self.z_dim)  #####
        self.use_l0 = hyperparams.use_l0

        self.sindy_coefficients = None

        # self.Weights = torch.nn.Parameter(torch.zeros(self.batch_size, out_shape))

        if hyperparams.learn_threshold_and_sigmoid:
            thresh_req_grad = True
        else:
            thresh_req_grad = False
        self.soft_threshold = nn.Parameter(torch.ones(self.library_dim, self.z_dim) * hyperparams.soft_threshold,
                                           requires_grad=thresh_req_grad)
        self.sigmoid_scale = nn.Parameter(torch.ones(self.library_dim, self.z_dim) * hyperparams.sigmoid_scale,
                                          requires_grad=thresh_req_grad)
        self.sig = nn.Sigmoid()

    def forward(self, x, x_dot, z, c, step, x_lib=None, device=0):

        x = x.type(torch.FloatTensor).to(device)
        if len(x.size()) < 2:
            x = x.unsqueeze(0).unsqueeze(1)
        elif len(x.size()) < 2:
            x = x.unsqueeze(0)

        if x_lib is None:
            x_lib = sindy_library(x, self.poly_order,
                                  include_constant=self.include_constant, include_sine=self.include_sine)
        else:
            x_lib = x_lib.type(torch.FloatTensor).to(device)

        l0_mask, pen = self.l0._get_mask()

        if self.autoencoder:
            mu, logvar = self.encoder(x, x_dot)
            sigma = logvar
            #sigma = torch.exp(0.5 * logvar)
            noisy_input = mu.to(device) + sigma*torch.randn(sigma.size()[1]).to(device)

        else:
            noisy_input = None; mu = None; logvar = None;

        sindy_coeffs, out_level1, out_level1_c, Weights, pen = self.get_masked_coefficients(z, c, step,  noisy_input,
                                                                                            shapes=self.shapes_no_bias,
                                                                                            len_seq=x.size(1),
                                                                                            batch_size=x.size(0),
                                                                                            l0_mask=l0_mask,
                                                                                            penalty=pen, device=device)

        self.sindy_coefficients = sindy_coeffs

        return self.dx(x_lib, sindy_coeffs), sindy_coeffs, out_level1, out_level1_c, Weights, pen, mu, logvar

    def dx(self, library, sindy_coeffs):

        b, t, lib_sz, zdim = sindy_coeffs.size()
        library = library.contiguous().view(-1, 1, lib_sz)
        sindy_coeffs = sindy_coeffs.contiguous().view(-1, lib_sz, zdim)

        return torch.bmm(library, sindy_coeffs).contiguous().view(b, t, zdim)

    def sample_coeffs(self, z, c, step,  noisy_input, shapes, len_seq, batch_size, device=0):
        if step == 0:
            z0 = None;
            c0 = None;
        else:
            z0 = z
            c0 = c

        zn, cn, Weights = self.hypernet1(len_seq, batch_size, z0, c0, network_input=None, device=device)

        """
        OUT = []
        for i in range(Weights.size(0)):
            #print("i with Weights", i)
            weights = Weights[i,:]
            #self.hypernet2.determine_Weights(shapes, weights)
            #out_level2 = self.hypernet2(batch_size=1, device=device) 
            out_level2 = self.hypernet2.forward_hypernet(shapes, weights)
            OUT.append(out_level2)
        """

        OUT = self.hypernet2.forward_hypernet(Weights, n=noisy_input, batch_size=batch_size * len_seq)#, hypernets=self.hypernets)
        OUT = OUT.view(batch_size, len_seq, OUT.size(1), OUT.size(2))

        return OUT, zn, cn, Weights

    def get_masked_coefficients(self, z, c, step, noisy_input, shapes, len_seq, batch_size, l0_mask, penalty, device=0):
        coefs, out_level1, out_level1_c, Weights = self.sample_coeffs(z, c, step,  noisy_input, shapes, len_seq, batch_size, device)

        if self.use_l0 == False:
            component = torch.abs(coefs).to(device) - self.soft_threshold.to(device)
            soft_mask = self.sig(self.sigmoid_scale * component).to(device)
            masked_coefficients = coefs.to(device) * soft_mask * self.threshold_mask.repeat(batch_size, len_seq, 1, 1)

            penalty = None
        else:

            if l0_mask is None:  #####
                l0_mask, penalty = self.l0._get_mask()

            masked_coefficients = coefs.to(device) * l0_mask.repeat(batch_size, len_seq, 1, 1).to(
                device) * self.threshold_mask.repeat(batch_size, len_seq, 1, 1)  #####

        """
        masked_coefficients = torch.zeros_like(sindy_coefficients)

        for i in range(len(sindy_coeffs)):
        masked_coefficients[i,:,:] = sindy_coeffs[i,:,:] * self.threshold_mask
        """

        return masked_coefficients, out_level1, out_level1_c, Weights, penalty

    def update_threshold_mask(self, threshold, Sindy_coeffs, device):
        # mask = self.get_masked_coefficients(device=device)
        mask = Sindy_coeffs.unsqueeze(0) #self.sindy_coefficients
        print("need to fix", mask.size())
        mask = torch.mean(mask, dim=0)
        print("again", mask.size())
        print("whoa", torch.mean(torch.abs(mask), dim=0).size())

        print((torch.mean(torch.abs(mask), dim=0) < threshold).sum())
        self.threshold_mask[torch.mean(torch.abs(mask), dim=0) < threshold] = 0

    def kl_old(self, sindy_coeffs):
        num_samples = sindy_coeffs.size(1)
        masked_coeffs = sindy_coeffs.reshape(num_samples, -1)  # 250 x 60
        gen_weights = masked_coeffs.transpose(1, 0)  # 60 x 250
        prior_samples = torch.randn_like(gen_weights)
        eye = torch.eye(num_samples, device=gen_weights.device)  # 250 x 250
        wp_distances = (prior_samples.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2  # 60 x 250 x 250
        ww_distances = (gen_weights.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2  # 60 x 250 x 250

        # zero out indices that were thresholded so kl isn't calculated for them
        # wp_distances = wp_distances * self.threshold_mask
        # ww_distances = ww_distances * self.threshold_mask

        wp_distances = torch.sqrt(torch.sum(wp_distances, 0) + 1e-8)  # 250 x 250
        wp_dist = torch.min(wp_distances, 0)[0]  # 250
        ww_distances = torch.sqrt(torch.sum(ww_distances, 0) + 1e-8) + eye * 1e10  # 250 x 250
        ww_dist = torch.min(ww_distances, 0)[0]  # 250

        # mean over samples
        kl = torch.mean(torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8))
        kl *= gen_weights.shape[0]
        kl += torch.log(torch.tensor(float(num_samples) / (num_samples - 1)))

        return kl

    def kl(self, mu, logvar):
        """Calculate the KL divergence.

        Calculates the KL divergence between q(mu, logvar) and the
        standard normal distribution with diagonal covariance.
        Assumes q is a normal distribution.

        Args:
            mu: A torch.tensor of shape (batch_size x z_dim) for the mean
                of q.
            logvar: A torch.tensor of shape (batch_size x z_dim) for the log
                of the variance of q.

        Returns:
            The calculated KL divergence as a torch.tensor of shape [batch_size].
        """
        print("muuuuuuu", mu.size(), logvar.size())
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)

    """
    def sparsify_sindy_dot(self, sindy_coeff_vec):

        sindy_coeff_mean = sindy_coeff_vec.mean(0).mean(1);

        lookup = (sindy_coeff_vec.max(0).values.max(0).values - sindy_coeff_vec.min(0).values.min(0) < 0.1 * sindy_coeff_mean)
        self.hypernet2.hypernet_mask[lookup] = 0

        if lookup.sum() != 0:
            self.net.hypernets = 2
    """