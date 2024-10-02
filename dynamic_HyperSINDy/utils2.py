import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from args_params_hyperparams import parse_hyperparams, parse_args
args = parse_args()
hyperparams = parse_hyperparams()

from utils1 import equation_sindy_library
def update_threshold_mask(net, threshold, threshold_increment, threshold_timer, epoch, device, beta, beta_max, folder):
    with torch.no_grad():

            if (epoch % threshold_timer == 0) and (epoch != 0):
                Sindy_coeffs = torch.load(folder + "/Sindy_coeffs_epoch" + str(epoch) + "_trial_" + str(9))

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


def eval_model(net, params, train_set, device, epoch):
    # sample trajectory
    z, x_dot_vec = sample_trajectory(net, device, train_set.x[0, :, :].numpy(),
                                     params.exp_batch_size, params.dt, params.exp_timesteps)

    # plot trajectory
    plot_trajectory(epoch, train_set.x.numpy(), z)

    # get equations
    t = 99
    equations = get_equations(net, params.exp_batch_size, device, params.z_dim, params.poly_order,
                              params.include_constant, params.include_sine, t)

    eq_mean = str(equations[1]) + "  \n" + str(equations[2]) + "  \n" + str(equations[3])
    eq_std = str(equations[5]) + "  \n" + str(equations[6]) + "  \n" + str(equations[7])
    draw_equations(epoch, equations, params["z_dim"])


# returns: batch_size x ts x z_dim
def sample_trajectory(net, device, x0, batch_size=10, dt=1e-2, ts=100):
    zc = torch.from_numpy(x0).type(torch.FloatTensor).to(device)
    # zc = torch.stack([zc for _ in range(batch_size)], dim=0)
    zc = zc.unsqueeze(1)

    zs = []

    z = None;
    step = 0;
    x_lib = None;

    sindy_coefficients = []
    x_dot_vec = []
    for i in range(ts):
        x_dot, sindy_coeffs, z, weights = net(zc, z, step, device=device)
        # print(sindy_coeffs[1,0:3,:5,:5])

        step += 1
        zc = zc + x_dot * dt
        zs.append(zc)
        sindy_coefficients.append(sindy_coeffs)

        x_dot_vec.append(x_dot)

    zs = torch.stack(zs, dim=0)
    zs = torch.transpose(zs, 0, 1).squeeze()
    sindy_coefficients = torch.transpose(torch.stack(sindy_coefficients, dim=0), 0, 1)
    net.sindy_coefficients = sindy_coefficients.squeeze()
    x_dot_vec = torch.stack(x_dot_vec)

    return zs.detach().cpu().numpy(), x_dot_vec


def plot_trajectory(epoch, z_true, z_pred, figsize=None):
    batch_size, T, z_dim = z_pred.shape

    if figsize is None:
        fig = plt.figure(figsize=(batch_size + 1, 3.5), dpi=300)
    else:
        fig = plt.figure(figsize=figsize, dpi=300)

    for i in range(batch_size + 1):
        if z_dim == 1:
            ax = fig.add_subplot(1, batch_size, i + 1)
            ax.plot(z_pred[i, :, 0][j], color='red', label='Pred')
            ax.plot(z_true, color='blue', label='GT')
            ax.legend(loc='best')
        elif z_dim == 2:
            ax = fig.add_subplot(1, batch_size, i + 1)
            ax.plot(z_pred[i, :, 0], color='red', label='X Pred')
            ax.plot(z_pred[i, :, 1], color='blue', label='Y Pred')
            ax.plot(z_true[:, 0], color='yellow', label='X GT')
            ax.plot(z_true[:, 1], color='green', label='Y GT')
            ax.legend(loc='best')
        elif z_dim == 3:
            ax = fig.add_subplot(1, batch_size + 1, i + 1, projection='3d')
            if i == 0:
                # plot the first trajectory
                ax.plot(z_true[:, 0, 0], z_true[:, 0, 1], z_true[:, 0, 2], color='red', label="GT")
            else:
                ax = fig.add_subplot(1, batch_size + 1, i + 1, projection='3d')
                ax.plot(z_pred[i - 1, :, 0], z_pred[i - 1, :, 1], z_pred[i - 1, :, 2], color='blue', label="Pred")
    fig.subplots_adjust(wspace=0.0, hspace=0.0)


def get_equations(net, len_seq, device, z_dim, poly_order, include_constant, include_sine, t):
    starts = ["X' = ", "Y' = ", "Z' = "]
    library = equation_sindy_library(z_dim, poly_order, include_constant=include_constant, include_sine=include_sine)

    equations = []
    sindy_coeffs = net.sindy_coefficients

    mean_coeffs, std_coeffs = sindy_coeffs_stats(sindy_coeffs)

    equations.append("MEAN at time t = {}".format(t))
    update_equation_list(equations, library, mean_coeffs, starts, z_dim, t)
    # if np.sum(np.abs(std_coeffs)) != 0:
    equations.append("STD at time t = {}".format(t))
    update_equation_list(equations, library, std_coeffs, starts, z_dim, t)

    return equations


def sindy_coeffs_stats(sindy_coeffs):
    coefs = sindy_coeffs.detach().cpu().numpy()

    if len(sindy_coeffs.size()) == 4:
        return np.mean(coefs, axis=0), np.std(coefs, axis=0)
    else:
        return coefs, 0


def update_equation_list(equations, library, coefs, starts, z_dim, t):
    for i in range(z_dim):
        equations.append(build_equation(library, coefs[:, :, i], starts[i], t))


def build_equation(lib, coef, eq, t):
    l = len(eq)
    for i in range(coef.shape[1]):
        if coef[t, i] != 0:
            if i == len(coef) - 1:
                eq += str(coef[t, i]) + lib[i]
            else:
                eq += str(coef[t, i]) + lib[i] + ' + '
    if eq[-2] == '+':
        eq = eq[:-3]

    if len(eq) == l:
        eq += " 0"

    return eq


def draw_equations(epoch, equations, z_dim):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.gca()
    x_pos = 0.25
    y_pos = 0.9
    dy = 0.055

    for idx, eq in enumerate(equations):
        if idx % (z_dim + 1) == 0:
            ax.text(x_pos, y_pos, eq, color='red', fontweight='bold')
        else:
            ax.text(x_pos, y_pos, eq)
        y_pos -= dy

    clear_ax_extra(ax)


def clear_ax_extra(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect('equal')


def smooth_function(sindy_coeff, smooth_interval):
    smoother = 1 / smooth_interval * torch.ones(1, 1, smooth_interval)

    sindy_coeff2 = torch.permute(sindy_coeff, (1, 2, 3, 0)).reshape(-1, 1, 10000)
    sindy_coeff_padded = F.pad(sindy_coeff2, (int(smooth_interval // 2), int(smooth_interval // 2)-1), "reflect")

    sindy_coeff_smooth = F.conv1d(sindy_coeff_padded, smoother, bias=None, stride=1)
    sindy_coeff_smooth = sindy_coeff_smooth.view(19, 3, 1, sindy_coeff_smooth.size(2)).squeeze()
    sindy_coeff_smooth = torch.permute(sindy_coeff_smooth, (2, 0, 1)).unsqueeze(0)

    return sindy_coeff_smooth