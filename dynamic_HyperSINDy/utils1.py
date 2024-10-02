import torch
import torch.nn as nn
from scipy.special import binom

import numpy as np

from args_params_hyperparams import parse_hyperparams, parse_args
args = parse_args()
hyperparams = parse_hyperparams()

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
    device = X.device
    l = library_size(n, poly_order, include_sine, include_mult_sine, include_constant)
    library = torch.ones((b, m, l), device=device)
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


def equation_sindy_library(n=3, poly_order=3, device=1, include_sine=False, include_mult_sine=False,
                           include_constant=True):
    # timesteps x latent dim
    l = library_size(n, poly_order, include_sine, include_constant)
    str_lib = []
    if include_constant:
        index = 1
        str_lib = ['']
    X = ['x', 'y', 'z']

    for i in range(n):
        str_lib.append(X[i])

    if poly_order > 1:
        for i in range(n):
            for j in range(i, n):
                str_lib.append(X[i] + X[j])

    if poly_order > 2:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    str_lib.append(X[i] + X[j] + X[k])

    if poly_order > 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        str_lib.append(X[i] + X[j] + X[k] + X[q])

    if poly_order > 4:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        for r in range(q, n):
                            str_lib.append(X[i] + X[j] + X[k] + X[q] + X[r])

    if include_sine:
        for i in range(n):
            str_lib.append('sin(' + X[i] + ')')

    if include_mult_sine:
        for i in range(n):
            str_lib.append(X[i] + 'sin(' + X[i] + ')')

    return str_lib


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


def decide_requires_grad(i, train_interval, net):
    """
    if i % (2 * train_interval) < train_interval:
        for n, p in net.named_parameters():
            if "D" in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
    else:
        for n, p in net.named_parameters():
            if "D" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
    """

    if i % train_interval < int(train_interval//3):
        for n, p in net.named_parameters():
            p.requires_grad = True
    elif (i % train_interval < int(2*train_interval//3)) and (i % train_interval > int(train_interval//3)):
        for n, p in net.named_parameters():
            if "D" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for n, p in net.named_parameters():
            if "D" in n:
                p.requires_grad = False
            else:
                p.requires_grad = True