import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import glob
import math
from scipy.special import binom

import argparse
import derivative

import pickle

torch.autograd.set_detect_anomaly(True)


def parse_hyperparams():
    hyperparams = {}
    hyperparams['dataset'] = "harmonic_osc"  # CHANGE
    # hyperparams['data_folder'] = '/home/doris/HyperSINDy/data/'
    # hyperparams['data_folder'] = '/gscratch/deepthought/dorisv/HyperSINDy/data/'
    hyperparams['data_folder'] = '/gscratch/dynamicsai/doris/HyperSINDy/data/sigmoid/' #sometime CHANGE
    #hyperparams['data_folder'] = '/gscratch/dynamicsai/doris/HyperSINDy/data/sinusoid/'
    #hyperparams['data_folder'] = '/gscratch/dynamicsai/doris/HyperSINDy/data/switch_signal/'
    # hyperparams['data_folder'] = '/gscratch/deepthought/dorisv/HyperSINDy/data/drift_diffusion/'
    # hyperparams['data_folder'] = '/home/doris/HyperSINDy/data/sinusoid/'
    #hyperparams['data_folder'] = '/home/doris/HyperSINDy/data/sigmoid/'  # sometimes CHANGE

    hyperparams['dt'] = 0.01
    hyperparams['noise_dim1'] = 25
    hyperparams['noise_dim2'] = 25
    hyperparams['hidden_dim1'] = 256
    hyperparams['hidden_dim2'] = 256
    hyperparams['batch_size'] = 1
    hyperparams['len_seq'] = 1
    hyperparams["len_seq2"] = 100
    hyperparams['learning_rate'] = 1e-3
    hyperparams['adam_reg'] = 1e-5

    hyperparams['weight_decay'] = None
    hyperparams['clip'] = 1.0
    hyperparams['use_l0'] = True
    hyperparams['weight_decay_l0'] = 1e-5
    hyperparams['beta'] = 1.0
    hyperparams['beta_init'] = 0.01
    hyperparams['beta_inc'] = None
    hyperparams['amsgrad'] = True
    hyperparams['epochs'] = 551
    hyperparams['random'] = True
    hyperparams['soft_threshold'] = 0.1
    hyperparams['sigmoid_scale'] = 1.0
    hyperparams['threshold'] = 0.1  # !!
    hyperparams['threshold_increment'] = 0.05
    hyperparams['threshold_interval'] = 5 # CHANGE
    hyperparams["threshold_convergence"] = 1e-5
    hyperparams["smooth_interval"] = 1000
    hyperparams['prior'] = "normal"
    hyperparams["use_all_ic"] = True
    hyperparams['model_choice'] = "mixed basis for hypernet"
    hyperparams["autoencoder"] = False
    hyperparams["learn_threshold_and_sigmoid"] = False
    return hyperparams


def parse_args():
    params = {}

    # base folders
    params["saved_model"] = False
    # params["path_model"] = "/gscratch/dynamicsai/doris/HyperSINDy/hello/saved_results/new_hyak/harmonic_osc_sigmoid_len_seq1/Sindy_coeffs_harmonic_osc_sigmoid_T0.1_Tincr0.05_TI5_A_lenseq_1_no_ae_short3"
    params["path_model"] = "/data/doris/harmonic_osc_sigmoid_len_seq10/Sindy_coeffs_harmonic_osc_sinusoid_T0.1_Tincr0.05_TI5_B_lenseq_10_ae_1trial_long_NEW_mlp_alphas_diff10"

    # sindy parameters
    params['z_dim'] = 2  # CHANGE
    params['poly_order'] = 3
    params['include_constant'] = False
    params['include_sine'] = False

    # training parameters
    params['gamma_factor'] = 0.999
    params['checkpoint_interval'] = 25

    # dataset parameters
    params['norm_data'] = False
    params['scale_data'] = 0.0

    # experiment parameters
    params['exp_batch_size'] = 10
    params['exp_timesteps'] = 100
    params['eval_interval'] = 1

    # other
    params['device'] = 1
    params['load_cp'] = 0
    params['print_folder'] = 1
    params['statistic_batch_size'] = 500

    params['mix_dim'] = 10

    params["seed"] = 1
    return params


args = parse_args()
hyperparams = parse_hyperparams()

if torch.cuda.is_available():
    # device = torch.cuda.device(3) #torch.device('cuda')
    # device = torch.device('cuda')
    device = torch.device('cpu')
else:
    device = torch.device('cpu')


def library_size(n, poly_order, use_sine=False, use_mult_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    if use_sine:
        l += n
    if use_mult_sine:
        l += n
    if not include_constant:
        l -= 1

    return l


def sindy_library(X, poly_order=3, include_sine=False, include_mult_sine=False, include_constant=True):
    # batch x latent dim
    b, m, n = X.shape
    print("b,m,n", b, m, n)
    device = X.device
    l = library_size(n, poly_order, include_sine, include_mult_sine, include_constant)
    library = torch.ones((b, m, l), device=device)
    print("library size", library.size())
    index = 0

    if include_constant:
        index = 1

    for i in range(n):
        library[:, :, index] = X[:, :, i]
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i, n):
                library[:, :, index] = X[:, :, i] * X[:, :, j]
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    library[:, :, index] = X[:, :, i] * X[:, :, j] * X[:, :, k]
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        library[:, :, index] = X[:, :, i] * X[:, :, j] * X[:, :, k] * X[:, :, q]
                        index += 1

    if poly_order > 4:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        for r in range(q, n):
                            library[:, :, index] = X[:, :, i] * X[:, :, j] * X[:, :, k] * X[:, :, q] * X[:, :, r]
                            index += 1

    if include_sine:
        for i in range(n):
            library[:, :, index] = torch.sin(X[:, :, i])
            index += 1

    if include_mult_sine:
        for i in range(n):
            library[:, :, index] = X[:, :, i] * torch.sin(X[:, :, i])
            index += 1

    return library


def load_data(hyperparams, params):
    fpath, data_folder, start_file = get_data_path(hyperparams["data_folder"], hyperparams["dataset"])
    #hyperparams["param"], hyperparams["drift"], hyperparams["diff"])

    if hyperparams["use_all_ic"]:

        datasets = []

        for file in os.listdir(data_folder):
            if file.startswith(start_file):
                datasets.append(SyntheticDataset(data_folder + "/" + file, params))

        if datasets != []:
            return integrate_datasets(datasets)
    else:
        return SyntheticDataset(fpath)


def str_filename(s):
    if len(s) == 1:
        s = s[0]
    else:
        s = str(s)

    return s


def get_sub_data_path(dataset):#, param, drift, diff):
    #param = str_filename(param)
    #drift = str_filename(drift)
    #diff = str_filename(diff)

    # CHANGE
    # file = "state-" + param + "_sinusoid_low_freq_sinx_allIC_harmonic_osc_v1"
    file = "state-B_allIC_harmonic_osc_short_v2" #sigmoid!
    # file = "state-A_allIC_harmonic_osc_v1"
    # file = "state-A_allIC_harmonic_osc_long_v2"
    #file = "state-A_B_allIC_harmonic_osc_v100"
    #file = "state-A_allIC_harmonic_osc"
    #file =  "state-A_allIC_harmonic_osc_v1" #"state-A_allIC_harmonic_osc_frequentSwitch" #"state-A_B_allIC_harmonic_osc_3D" #"state-A_allIC_harmonic_osc" #"state-A_B_allIC_harmonic_osc"  #"state-A_allIC_harmonic_osc_v0" #state-A_B_allIC_harmonic_osc_high_freq_v0

    return dataset + "/" + file + "/", file


def get_data_path(data_folder, dataset):#, param, drift, diff):
    path, file = get_sub_data_path(dataset)#, param, drift, diff)
    path = data_folder + path
    return path, data_folder + dataset, file


class SyntheticDataset(Dataset):

    def __init__(self, fpath, params):
        self.x = torch.from_numpy(np.load(fpath + "/x_train.npy"))
        if len(self.x.size()) == 2:
            self.x = self.x.unsqueeze(0)
        # self.x_dot = torch.from_numpy(np.load(fpath + "x_dot.npy"))
        self.x_dot = self.fourth_order_diff(self.x, hyperparams["dt"])
        self.x_lib = sindy_library(
            self.x, params["poly_order"],
            include_constant=params["include_constant"],
            include_sine=params["include_sine"])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.x_lib[idx], self.x_dot[idx]

    def fourth_order_diff(self, x, dt):
        dx = torch.zeros(x.size())
        dx[:, 0, :] = (-11.0 / 6) * x[:, 0, :] + 3 * x[:, 1, :] - 1.5 * x[:, 2, :] + x[:, 3, :] / 3
        dx[:, 1, :] = (-11.0 / 6) * x[:, 1, :] + 3 * x[:, 2, :] - 1.5 * x[:, 3, :] + x[:, 4, :] / 3
        dx[:, 2:-2, :] = (-1.0 / 12) * x[:, 4:, :] + (2.0 / 3) * x[:, 3:-1, :] - (2.0 / 3) * x[:, 1:-3, :] + (
                1.0 / 12) * x[:, :-4, :]
        dx[:, -2, :] = (11.0 / 6) * x[:, -2, :] - 3.0 * x[:, -3, :] + 1.5 * x[:, -4, :] - x[:, -5, :] / 3.0
        dx[:, -1, :] = (11.0 / 6) * x[:, -1, :] - 3 * x[:, -2, :] + 1.5 * x[:, -3, :] - x[:, -4, :] / 3.0

        return dx / dt


class MultipleSyntheticDatasets(Dataset):

    def __init__(self, x, x_dot, x_lib):
        self.x = x
        self.x_dot = x_dot
        self.x_lib = x_lib

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # print("sizes", self.x.size(), self.x_lib.size(), self.x_dot.size())
        return self.x[idx], self.x_lib[idx], self.x_dot[idx]


def integrate_datasets(datasets):
    x = datasets[0].x;
    x_dot = datasets[0].x_dot;
    x_lib = datasets[0].x_lib;

    for j in range(1, len(datasets)):
        x = torch.cat((x, datasets[j].x), dim=0)
        x_dot = torch.cat((x_dot, datasets[j].x_dot), dim=0)
        x_lib = torch.cat((x_lib, datasets[j].x_lib), dim=0)

    return MultipleSyntheticDatasets(x, x_dot, x_lib)


train_set = load_data(hyperparams, args)

# this worked OK
train_set.x = train_set.x[0, :2000, :].unsqueeze(0)
train_set.x_dot = train_set.x_dot[0, :2000, :].unsqueeze(0)
train_set.x_lib = train_set.x_lib[0, :2000, :].unsqueeze(0)

T = train_set.x.size()[1];
dt = hyperparams["dt"]
batch = train_set.x.size()[0];
n_dim = train_set.x.size()[2];
x_vec = train_set.x.permute(0, 2, 1).reshape(-1, T)

differentiator = derivative.Kalman(alpha=0.5)
x_smooth = torch.from_numpy(differentiator.x(x_vec, np.arange(0, dt * T, dt))).permute(1, 0).reshape(batch, T, n_dim)
# x_smooth = torch.from_numpy(differentiator.x(x_vec, np.arange(0,dt*T, dt)[:-1])).permute(1,0).unsqueeze(0)
x_dot_smooth = torch.from_numpy(differentiator.d(x_vec, np.arange(0, dt * T, dt))).permute(1, 0).reshape(batch, T,
                                                                                                         n_dim)
# x_dot_smooth = torch.from_numpy(differentiator.d(x_vec, np.arange(0,dt*T, dt)[:-1])).permute(1,0).unsqueeze(0)
x_lib_smooth = sindy_library(x_smooth, 3, include_constant=False)

print("x_smooth size", x_smooth.size())
train_set.x = x_smooth
train_set.x_dot = x_dot_smooth
train_set.x_lib = x_lib_smooth

trainloader = DataLoader(train_set, batch_size=hyperparams["batch_size"],
                         shuffle=False, drop_last=True)

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight)
    elif isinstance(layer, nn.LayerNorm):
        layer.bias.data.fill_(0.01)
    elif isinstance(layer, nn.BatchNorm1d):
        layer.bias.data.fill_(0.01)

def get_weight_dim(net):
    out_shape = []
    out_shape_no_bias = []

    for i in range(len(net.layers)):
        network = net.layers[i]

        outs = []
        for n, p in network.named_parameters():
            if p.size() != []:
                outs.append(p.size())
                if len(p.size()) == 2:
                    out_shape_no_bias.append(p.size())

        if outs != []:
            out_shape.append(outs)

    return out_shape, out_shape_no_bias


def prod_dim(out_shape):
    out_size = 0;
    all_sizes = []
    for o1 in out_shape:
        for o2 in o1:
            out_size += np.prod(o2)
            all_sizes.append(np.prod(o2))

    return out_size, all_sizes


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
            penalty = self.sig(self.loc - self.temp * self.gamma_zeta_ratio).sum()  # .sum(1).mean()
        else:
            s = self.sig(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return hard_sigmoid(s), penalty


def smooth_function(sindy_coeff, smooth_interval, device):
    smoother = 1 / smooth_interval * torch.ones(1, 1, smooth_interval)

    sindy_coeff2 = torch.permute(sindy_coeff, (1, 2, 3, 0)).reshape(-1, 1, 10000)
    sindy_coeff_padded = F.pad(sindy_coeff2, (int(smooth_interval // 2), int(smooth_interval // 2) - 1), "reflect")

    sindy_coeff_smooth = F.conv1d(sindy_coeff_padded.to(device), smoother.to(device), bias=None, stride=1)
    sindy_coeff_smooth = sindy_coeff_smooth.view(9, 2, 1, sindy_coeff_smooth.size(2)).squeeze()
    sindy_coeff_smooth = torch.permute(sindy_coeff_smooth, (2, 0, 1)).unsqueeze(0)

    return sindy_coeff_smooth


def smoothness_reg(sindy_coeffs):
    norm = sindy_coeffs.size(0) * sindy_coeffs.size(1) * sindy_coeffs.size(2) * sindy_coeffs.size(3)
    sindy_coeffs_dt = ((sindy_coeffs[:, 1:, :, :] - sindy_coeffs[:, :-1, :, :]) ** 2).sum()

    return sindy_coeffs_dt


def make_folder(name):
    if not os.path.isdir(name):
        os.makedirs(name)


class Encoder(nn.Module):
    def __init__(self, hyperparams, args, out_dim, hidden_dim=64, bias=True, activation=nn.ELU(),
                 norm='batch'):
        super(Encoder, self).__init__()

        self.in_dim = 2 * args["z_dim"]
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

        # self.fc1 = nn.Linear(self.out_dim, int(self.out_dim//2))
        # self.fc2 = nn.Linear(self.out_dim, int(self.out_dim//2))

    def forward(self, x, x_dot):
        x = x.view(-1, x.size()[-1]);
        x_dot = x_dot.view(-1, x_dot.size()[-1])
        print(x.size(), x_dot.size())
        x_vec = torch.cat((x, x_dot), dim=1)
        out = self.encoder_layers(x_vec)

        print("out size", out.size())
        return out[:, :int(self.out_dim // 2)], out[:, int(self.out_dim // 2):]
        # return self.fc1(out), self.fc2(out)


class HyperNet(nn.Module):
    def __init__(self, in_dim, out_shape, random=True, hidden_dims=[8, 16, 32], bias=True, activation=nn.ELU(),
                 norm='batch'):
        super(HyperNet, self).__init__()

        self.in_dim = in_dim
        self.out_shape = out_shape
        self.number_of_layers = len(hidden_dims)
        self.random = random

        in_features = self.in_dim
        layers = [];
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

        # self.hypernet_mask = nn.Parameter(torch.zeros(self.out_shape), requires_grad = False)
        # self.hypernet_mask[0,0] = 1; self.hypernet_mask[0,1] = 1;

    def forward_hypernet(self, Weights, n=None, batch_size=1, device="cpu"):
        if n is None:
            if self.random:
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

            #not working... very big sindy coeff values!
            #self.layers[self.layer_numbers[i]].weight = torch.nn.Parameter(weight.mean(0))
            #self.layers[self.layer_numbers[i]].bias = torch.nn.Parameter(bias.mean(0))

            x = torch.bmm(weight.to(device), x.to(device)).squeeze()
            if len(x.size()) < 2:
                x = x.unsqueeze(0)
            if bias != None:
                x += bias.to(device)

            # batch norm
            # n_chans = x.shape[0]
            # m = torch.nn.BatchNorm1d(n_chans)
            # x = m(x.permute(1,0)).permute(1,0)

            # if i >= 2:
            #    x = F.dropout(x, p=0.2)

            if i < len(Weights) - 1:
                x = F.elu(x, inplace=False)

        # x = x * self.hypernet_mask.repeat(batch_size, 1, 1).view(batch_size, -1).to(device)

        return x.reshape(n.size(0), *self.out_shape)


class HyperNet0(nn.Module):
    def __init__(self, basis, out_shape, len_seq, batch_size, mix_dim, device=0):
        super(HyperNet0, self).__init__()

        self.len_seq = len_seq
        self.mix_dim = mix_dim

        self.basis = basis
        self.out_shape = out_shape

        layers = [nn.Linear(25, 10), nn.ELU(), nn.Linear(10, 10), nn.ELU(), nn.Linear(10, self.out_shape)]
        self.mlp = nn.Sequential(*layers)

        self.weights = nn.Parameter(torch.randn((len_seq*batch_size), 25), requires_grad=False)

    def forward(self, input, len_seq, batch_size, device='cpu'):

        Weights = []
        print("this sucks")
        print("input size", input.size())
        weights = self.mlp(input)

        #self.weights.detach().to(device)
        for i in range(len(self.basis)):
            w = weights[:, i * self.mix_dim:(i + 1) * self.mix_dim]
            #w = w/w.sum();
            w_dim1, w_dim2 = self.basis[i].size(1), self.basis[i].size(2)
            #self.basis[i].detach().to(device)
            Weights.append(torch.matmul(w.view(-1, self.mix_dim), self.basis[i].view(self.mix_dim, -1)).view(batch_size * len_seq,
                                                                                                  w_dim1, w_dim2))
        return self.weights, Weights


class Net(nn.Module):
    def __init__(self, args, hyperparams):
        super(Net, self).__init__()

        self.z_dim = args["z_dim"]
        self.poly_order = args["poly_order"]

        self.include_constant = args["include_constant"]
        self.include_sine = args["include_sine"]
        self.statistic_batch_size = args["statistic_batch_size"]
        self.batch_size = hyperparams["batch_size"]
        self.len_seq = hyperparams["len_seq"]  # hyperparams["len_seq"]
        self.len_seq2 = hyperparams["len_seq2"]

        self.noise_dim1 = hyperparams["noise_dim1"]
        self.noise_dim2 = hyperparams["noise_dim2"]
        self.hypernet_hidden_dim2 = hyperparams["hidden_dim2"]

        self.random = hyperparams["random"]

        self.library_dim = library_size(self.z_dim, self.poly_order,
                                        include_constant=self.include_constant, use_sine=self.include_sine)

        self.mix_dim = args["mix_dim"]

        self.autoencoder = hyperparams["autoencoder"]
        if self.autoencoder:
            self.encoder = Encoder(hyperparams, args, 2 * hyperparams["noise_dim2"])

        self.hypernet2 = HyperNet(self.noise_dim2, (self.library_dim, self.z_dim),
                                  self.random, [self.hypernet_hidden_dim2 for _ in range(4)])
        out_shape, all_sizes = prod_dim(get_weight_dim(self.hypernet2)[0])
        self.shapes, self.shapes_no_bias = get_weight_dim(self.hypernet2)
        self.out_shape = out_shape

        self.basis = []
        out_shape_rnn = 0
        for i in range(len(self.shapes)):
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

        self.hypernet1 = HyperNet0(basis=self.basis, out_shape=out_shape_rnn,
                                   len_seq=self.len_seq, mix_dim=self.mix_dim, batch_size=self.batch_size)

        self.threshold_mask = nn.Parameter(torch.ones(self.library_dim, self.z_dim),
                                           requires_grad=False)

        self.l0 = _L0Norm(self.library_dim, self.z_dim)  #####
        self.use_l0 = hyperparams["use_l0"]

        self.sindy_coefficients = None

        if hyperparams["learn_threshold_and_sigmoid"]:
            thresh_req_grad = True
        else:
            thresh_req_grad = False
        self.soft_threshold = nn.Parameter(torch.ones(self.library_dim, self.z_dim) * hyperparams["soft_threshold"],
                                           requires_grad=thresh_req_grad)
        self.sigmoid_scale = nn.Parameter(torch.ones(self.library_dim, self.z_dim) * hyperparams["sigmoid_scale"],
                                          requires_grad=thresh_req_grad)
        self.sig = nn.Sigmoid()

    def forward(self, input, x, x_dot, x_lib=None, device=0):

        x = x.type(torch.FloatTensor).to(device)
        if len(x.size()) < 2:
            x = x.unsqueeze(0).unsqueeze(1)

        if x_lib is None:
            x_lib = sindy_library(x, self.poly_order,
                                  include_constant=self.include_constant, include_sine=self.include_sine)
        else:
            x_lib = x_lib.type(torch.FloatTensor).to(device)

        l0_mask, pen = self.l0._get_mask()

        if self.autoencoder:
            mu, logvar = self.encoder(x, x_dot)
            #sigma = logvar
            sigma = torch.exp(0.5 * logvar)
            noisy_input = mu.to(device) + sigma * torch.randn(sigma.size()[1]).to(device)

        else:
            noisy_input = None;
            mu = None;
            logvar = None;

        sindy_coeffs, weights, Weights, pen = self.get_masked_coefficients(noisy_input, input,
                                                                            shapes=self.shapes_no_bias,
                                                                            len_seq=x.size(1),
                                                                            batch_size=x.size(0),
                                                                            l0_mask=l0_mask,
                                                                            penalty=pen, device=device)

        self.sindy_coefficients = sindy_coeffs

        return self.dx(x_lib, sindy_coeffs), sindy_coeffs,  weights, Weights, pen, mu, logvar

    def dx(self, library, sindy_coeffs):

        b, t, lib_sz, zdim = sindy_coeffs.size()
        # print("library and sindy_coefficients sizes", library.size(), sindy_coeffs.size())
        library = library.contiguous().view(-1, 1, lib_sz)
        sindy_coeffs = sindy_coeffs.contiguous().view(-1, lib_sz, zdim)

        return torch.bmm(library, sindy_coeffs.float()).contiguous().view(b, t, zdim)

    def sample_coeffs(self, noisy_input, input, shapes, len_seq, batch_size, device=0):

        weights, Weights = self.hypernet1(input, len_seq, batch_size, device=device)

        OUT = self.hypernet2.forward_hypernet(Weights, n=noisy_input, batch_size=batch_size * len_seq)
        OUT = OUT.view(batch_size, len_seq, OUT.size(1), OUT.size(2))

        return OUT, weights, Weights

    def get_masked_coefficients(self, noisy_input, input, shapes, len_seq, batch_size, l0_mask, penalty, device=0):
        coefs, weights, Weights = self.sample_coeffs(noisy_input, input, shapes, len_seq, batch_size, device)

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
        return masked_coefficients, weights, Weights, penalty

    def update_threshold_mask(self, threshold, Sindy_coeffs, device):
        # mask = self.get_masked_coefficients(device=device)
        if len(Sindy_coeffs.size()) == 3:
            Sindy_coeffs = Sindy_coeffs.unsqueeze()

        mask = Sindy_coeffs  # self.sindy_coefficients
        mask = torch.mean(mask, dim=0)

        print("!!!!", Sindy_coeffs.size(), mask.size(), torch.mean(torch.abs(mask), dim=0).size(),
              self.threshold_mask.size())
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


net = Net(args, hyperparams).to(device).float()
net.apply(init_weights)

if args["saved_model"]:
    print("I'm loading model")
    path = args["path_model"] + "/model_epoch" + str(199) + "_trial_" + str(0) + ".pt"
    net.load_state_dict(torch.load(path))
else:
    print("I'm not loading model")

input = 0*nn.Parameter(0.01*torch.randn(1,25).float(), requires_grad = True)
input.retain_grad()
#optim.param_groups.append({'params': input})

#optim = torch.optim.AdamW(
#    list(net.parameters()) + [input], lr=hyperparams["learning_rate"],
#    weight_decay=hyperparams["adam_reg"],
#    amsgrad=hyperparams["amsgrad"])

optim = torch.optim.Adam(
    list(net.parameters()) + [input], lr=hyperparams["learning_rate"], weight_decay=hyperparams["adam_reg"])

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optim, gamma=args["gamma_factor"])

optim2 = torch.optim.AdamW(
    net.parameters(), lr=hyperparams["learning_rate"],
    weight_decay=hyperparams["adam_reg"],
    amsgrad=hyperparams["amsgrad"])

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optim2, gamma=args["gamma_factor"])

weight_decay_l0 = hyperparams["weight_decay_l0"]  #####
use_l0 = hyperparams["use_l0"]  #####


def update_threshold_mask(net, threshold, threshold_increment, threshold_timer, epoch, ii, device, beta, beta_max,
                          folder):
    with torch.no_grad():
        if (epoch % threshold_timer == 0) and (epoch != 0):
            Sindy_coeffs = torch.load(folder + "/Sindy_coeffs_epoch" + str(epoch) + "_trial_" + str(ii))

            print("I'm here 1, epoch, beta", epoch, beta)
            if (beta == beta_max):
                net.update_threshold_mask(threshold, Sindy_coeffs, device)
                print("new threshold", threshold + threshold_increment)
                return threshold + threshold_increment
        return threshold


def update_beta(beta, beta_increment, beta_max):
    beta += beta_increment
    print("new beta", beta)
    if beta > beta_max:
        beta = beta_max
    return beta


epochs = hyperparams["epochs"]
beta = hyperparams["beta"]
if hyperparams["beta_inc"] is None:
    hyperparams["beta_inc"] = hyperparams["beta"] / 5
threshold_convergence = hyperparams["threshold_convergence"]

#folder = "/data/doris/harmonic_osc_sigmoid_len_seq" + str(
#    hyperparams["len_seq"]) + "/Sindy_coeffs_harmonic_osc_sinusoid_T" + str(hyperparams["threshold"]) + "_Tincr" + str(
#    hyperparams["threshold_increment"]) + "_TI" + str(hyperparams["threshold_interval"]) + "_B_lenseq_" + str(
#    hyperparams["len_seq"]) + "_ae_1trial_long_NEW_mlp_alphas_diff10"  # CHANGE

folder = "saved_results/new_hyak/harmonic_osc_sigmoid_len_seq" + str(
    hyperparams["len_seq"]) + "/Sindy_coeffs_harmonic_osc_sigmoid_T" + str(hyperparams["threshold"]) + "_Tincr" + str(
    hyperparams["threshold_increment"]) + "_TI" + str(hyperparams["threshold_interval"]) + "_B_lenseq_" + str(
    hyperparams["len_seq"]) + "_ae_no_kld_1trial_short_0input_NEW_mlp_alternateLearning"  # CHANGE
# folder = "saved_results/new_hyak/harmonic_osc_sigmoid_len_seq" + str(hyperparams["len_seq"]) + "/Sindy_coeffs_harmonic_osc_sigmoid_T0.05_Tincr0.01_TI100_A_lenseq_10_short2"
make_folder(folder)


Loss = []
Recon = []
recon_smoothed = None

for epoch in range(200):  # CHANGE

    Sindy_coeffs22 = []

    print("!! EPOCH", epoch)
    net.train()
    recons, klds = 0, 0

    for ii, (x, x_lib, x_dot) in enumerate(trainloader):

        Sindy_coeffs_int22 = []
        ALPHAS = [];

        # torch.cuda.empty_cache()

        x = x.permute(1, 0, 2);
        x_dot = x_dot.permute(1, 0, 2);
        x_lib = x_lib.permute(1, 0, 2);
        x_saved = x.detach().clone();
        x_dot_saved = x_dot.detach().clone();
        x_lib_saved = x_lib.detach().clone();

        train_data_new = MultipleSyntheticDatasets(x, x_dot, x_lib)
        train_data_new_loader = DataLoader(train_data_new, batch_size=hyperparams["len_seq"],
                                           shuffle=False, drop_last=True)

        optim = torch.optim.AdamW(
            net.parameters(), lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["adam_reg"],
            amsgrad=hyperparams["amsgrad"])

        for i, (x, x_lib, x_dot) in enumerate(train_data_new_loader):
            print("epoch", epoch)
            print("len seq", hyperparams["len_seq"])

            print("ii", ii)
            print("i", i)

            x = x.permute(1, 0, 2);
            x_dot = x_dot.permute(1, 0, 2);
            x_lib = x_lib.permute(1, 0, 2);
            print("some sizes4", x.size(), x_lib.size(), x_dot.size())
            x_dot = x_dot.type(torch.FloatTensor).to(device)

            #if i > 0:
            #    net.mlp_input = nn.Parameter(MLP_input[i-1, :, :], requires_grad=True)
            x_dot_pred, sindy_coeffs, alphas, input_to_hypernet, reg, mu, logvar = net(input, x.to(device), x_dot.to(device), x_lib.unsqueeze(0).to(device), device=device)
            recon = ((x_dot_pred - x_dot) ** 2).sum(2).mean() + torch.norm(net.hypernet1.weights, 2) #+ torch.norm(alphas, "fro")

            #print("simple recon, reg norm", ((x_dot_pred - x_dot) ** 2).sum(2).mean(), torch.norm(alphas, "fro"))
            print("alphas", alphas)
            kld = 0#net.kl(mu, logvar).mean()
            print("kld", kld)

            loss = recon

            optim.zero_grad()
            #net.hypernet1.weights.retain_grad()
            #net.D_basis1.retain_grad(); net.D_basis2.retain_grad(); net.D_basis3.retain_grad();
            #net.D_basis4.retain_grad(); net.D_basis5.retain_grad();
            loss.backward(retain_graph=True)
            #loss.backward()

            if hyperparams["clip"] is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), hyperparams["clip"])

            optim.step()

            Sindy_coeffs_int22.append(net.sindy_coefficients)
            Loss.append(loss.item())
            Recon.append(recon.item())
            ALPHAS.append(alphas)

            print("sindy_coefficients", sindy_coeffs.size(), sindy_coeffs[:, :, :, :].mean(0).mean(0))

            if epoch == 199 and i==100:
                torch.save(net.state_dict(), folder + "/model_epoch" + str(epoch) + "_trial_" + str(ii) + "_i100.pt")

        ALPHAS = torch.stack(ALPHAS)

        Sindy_coeffs_int22 = torch.stack(Sindy_coeffs_int22).squeeze()
        Sindy_coeffs22 = Sindy_coeffs_int22.reshape(int(T // hyperparams['len_seq']), hyperparams['len_seq'],
                                                    net.library_dim, net.z_dim)

        torch.save(Sindy_coeffs22, folder + "/Sindy_coeffs_epoch" + str(epoch) + "_trial_" + str(ii))
        torch.save(net.state_dict(), folder + "/model_epoch" + str(epoch) + "_trial_" + str(ii) + ".pt")
        torch.save(ALPHAS, folder + "/ALPHAS_input_epoch" + str(epoch) + "_trial_" + str(ii))

        del Sindy_coeffs_int22, Sindy_coeffs22

    hard_threshold = update_threshold_mask(net, hyperparams["threshold"], hyperparams["threshold_increment"],
                                           hyperparams["threshold_interval"], epoch, ii, device,
                                           beta, hyperparams["beta"], folder)
    scheduler.step()
    beta = update_beta(beta, hyperparams["beta_inc"], hyperparams["beta"])
"""
net.load_state_dict(torch.load("saved_results/new_hyak//harmonic_osc_sigmoid_len_seq1/Sindy_coeffs_harmonic_osc_sigmoid_T0.1_Tincr0.1_TI5_B_lenseq_1_ae_1trial_short_NEW_mlp_alternateLearning/model_epoch199_trial_0_i100.pt"))
#/mmfs1/gscratch/dynamicsai/doris/HyperSINDy/hello/saved_results/new_hyak/harmonic_osc_sigmoid_len_seq1/Sindy_coeffs_harmonic_osc_sigmoid_T0.1_Tincr0.1_TI5_B_lenseq_1_ae_1trial_short_NEW_mlp_alternateLearning

for epoch in range(200):  # CHANGE

    Sindy_coeffs22 = []

    print("!! EPOCH", epoch)
    recons, klds = 0, 0

    for ii, (x, x_lib, x_dot) in enumerate(trainloader):
        net.train()

        Sindy_coeffs_int22 = []
        Loss = []; Recon = [];
        MLP_input = [];

        # torch.cuda.empty_cache()
        print("iii", ii)

        x = x.permute(1, 0, 2);
        x_dot = x_dot.permute(1, 0, 2);
        x_lib = x_lib.permute(1, 0, 2);
        x_saved = x.detach().clone();
        x_dot_saved = x_dot.detach().clone();
        x_lib_saved = x_lib.detach().clone();

        train_data_new = MultipleSyntheticDatasets(x, x_dot, x_lib)
        train_data_new_loader = DataLoader(train_data_new, batch_size=hyperparams["len_seq"],
                                           shuffle=False, drop_last=True)

        # train mlp_input
        for n, p in net.named_parameters():
            #if "weights" in n:
            #    p.requires_grad = True
            #else:
            p.requires_grad = False

        for n, p in net.named_parameters():
            print(n, p.requires_grad)

        for i, (x, x_lib, x_dot) in enumerate(train_data_new_loader):
            print("epoch", epoch)
            print("len seq", hyperparams["len_seq"])

            print("ii", ii)
            print("i", i)

            x = x.permute(1, 0, 2);
            x_dot = x_dot.permute(1, 0, 2);
            x_lib = x_lib.permute(1, 0, 2);
            print("some sizes4", x.size(), x_lib.size(), x_dot.size())
            x_dot = x_dot.type(torch.FloatTensor).to(device)

            #if epoch>0:
            #    print("?!", net.hypernet1.weights.size())
            #    net.hypernet1.weights = nn.Parameter(MLP_input_prev[i * net.len_seq:(i + 1) * net.len_seq, :, :].view(1,25), requires_grad=True)

            x_dot_pred, sindy_coeffs, alphas, input_to_hypernet, reg, mu, logvar = net(input, x.to(device), x_dot.to(device), x_lib.unsqueeze(0).to(device), device=device)
            print("intial recon",  ((x_dot_pred - x_dot) ** 2).sum(2).mean())
            recon = ((x_dot_pred - x_dot) ** 2).sum(2).mean() + torch.norm(input, 2)

            print("sindy_coefficients", sindy_coeffs.size(), torch.abs(sindy_coeffs[:, :, :,:]).mean(0).mean(0))
            print("recon", recon, torch.norm(input, 2))

            kld = 0  # net.kl(mu, logvar).mean()
            print("kld", kld)

            loss = recon

            optim.zero_grad()
            loss.backward(retain_graph=True)
            #loss.backward()


            if hyperparams["clip"] is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), hyperparams["clip"])

            optim.step()
            #print("increment", input.grad, 0.0001*hyperparams["learning_rate"]*input.grad)
            #input = input - hyperparams["learning_rate"]*input.grad
            #input.retain_grad()

            print("weights, I mean input", input)
            MLP_input.append(input)

        MLP_input = torch.stack(MLP_input)

        # train other params: encode, decode, rHyperNet, D_i tensors
        permuted_ind = torch.randperm(T)#np.array(range(T))
        train_data_new = MultipleSyntheticDatasets(x_saved[permuted_ind,:,:], x_dot_saved[permuted_ind,:,:], x_lib_saved[permuted_ind,:,:])
        train_data_new_loader = DataLoader(train_data_new, batch_size=hyperparams["len_seq2"],
                                           shuffle=False, drop_last=True)
        MLP_input_perm = MLP_input[permuted_ind,:,:]
        MLP_input_prev = MLP_input.clone()
        MLP_input_perm = nn.Parameter(MLP_input_perm, requires_grad = False)

        for n, p in net.named_parameters():
            if "weights" in n or "threshold_mask" in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        for n, p in net.named_parameters():
            print(n, p.requires_grad)

        for i, (x, x_lib, x_dot) in enumerate(train_data_new_loader):

            x = x.permute(1, 0, 2);
            x_dot = x_dot.permute(1, 0, 2);
            x_lib = x_lib.permute(1, 0, 2);
            print("some sizes4", x.size(), x_lib.size(), x_dot.size())

            x_dot = x_dot.type(torch.FloatTensor).to(device)

            #net.hypernet1.weights = nn.Parameter(MLP_input[i*net.len_seq2:(i+1)*net.len_seq2,:,:].squeeze(), requires_grad = False)
            x_dot_pred, sindy_coeffs, alphas, input_to_hypernet, reg, mu, logvar = net(MLP_input_perm.squeeze()[i*net.len_seq2:(i+1)*net.len_seq2,:], x.to(device), x_dot.to(device),
                                                                                       x_lib.unsqueeze(0).to(device), device=device)

            print("sindy_coefficients", sindy_coeffs.size(), torch.abs(sindy_coeffs[:, :, :,:]).mean(0).mean(0))

            recon = ((x_dot_pred - x_dot) ** 2).sum(2).mean()
            kld = 0#net.kl(mu, logvar).mean()
            print("kld", kld)

            if use_l0:
                loss = recon + weight_decay_l0 * reg + kld*beta
            else:
                loss = recon

            optim2.zero_grad()
            loss.backward(retain_graph=True)

            # loss.backward()
            if hyperparams["clip"] is not None:
                nn.utils.clip_grad_norm_(net.parameters(), hyperparams["clip"])

            optim2.step()

            Sindy_coeffs_int22.append(net.sindy_coefficients)
            Loss.append(loss.item())
            Recon.append(recon.item())

        Sindy_coeffs_int22 = torch.stack(Sindy_coeffs_int22).squeeze()
        Sindy_coeffs22 = Sindy_coeffs_int22.reshape(int(T // hyperparams['len_seq2']), hyperparams['len_seq2'],
                                                    net.library_dim, net.z_dim)

        torch.save(Sindy_coeffs22, folder + "/Sindy_coeffs_epoch" + str(epoch) + "_trial_" + str(ii) + "_part2")
        torch.save(net.state_dict(), folder + "/model_epoch" + str(epoch) + "_trial_" + str(ii) + "_part2.pt")
        torch.save(MLP_input, folder + "/MLP_input_epoch" + str(epoch) + "_trial_" + str(ii) + "_part2")

        net.hypernet1.weights = nn.Parameter(
            net.hypernet1.weights[:net.len_seq, :].view(1, 25), requires_grad=False)

        del Sindy_coeffs_int22, Sindy_coeffs22

    hard_threshold = update_threshold_mask(net, hyperparams["threshold"], hyperparams["threshold_increment"],
                                           hyperparams["threshold_interval"], epoch, ii, device,
                                           beta, hyperparams["beta"], folder)
    scheduler.step()
    beta = update_beta(beta, hyperparams["beta_inc"], hyperparams["beta"])
"""

with open('hyperparams.pkl', 'wb') as fp:
    pickle.dump(hyperparams, fp)

with open('args.pkl', 'wb') as fp:
    pickle.dump(args, fp)









