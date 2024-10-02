import torch
from torch.utils.data import Dataset
import numpy as np

from utils1 import sindy_library, equation_sindy_library
import os

from args_params_hyperparams import parse_hyperparams, parse_args
args = parse_args()
hyperparams = parse_hyperparams()


def load_data(hyperparams, params):
    fpath, data_folder, start_file = get_data_path(hyperparams.data_folder, hyperparams.dataset1, hyperparams.dataset2, hyperparams.data_file,
                                                   hyperparams.param, hyperparams.drift, hyperparams.diff)

    if hyperparams.use_all_ic:

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


def get_sub_data_path(dataset1, dataset2, file, param, drift, diff):
    param = str_filename(param)
    drift = str_filename(drift)
    diff = str_filename(diff)

    if file is None:
        # file = "state-" +  param + "_drift-" + str(drift) + "_diff-" + str(diff) + "_allIC_v3"
        #file = "state-" + param + "_drift-" + str(drift) + "_diff-" + str(diff) + "_allIC_osc_v2"
        #file = "state-" + param + "_allIC_lorenz_finalSINDy"
        #file = "2pca_neural_act_worm4_short_283_147_5to6to5.npy" #"2pca_neural_act_worm4.npy"
        #file = "state-s_allIC_lorenz_v1" #for sigmoid
        #file = "state-s_allIC_lorenz_v0" #for sinusoid
        file = "state-" + param + "_allIC_lorenz_long_v2"

    print(dataset1 + "/" + dataset2 + "/" + file + "/", file)
    return dataset1 + "/" + dataset2 + "/" + file + "/", file


def get_data_path(data_folder, dataset1, dataset2, datafile, param, drift, diff):
    path, file = get_sub_data_path(dataset1, dataset2, datafile, param, drift, diff)
    path = data_folder + path
    print(path, data_folder + "/" + dataset1, dataset2, file)
    return path, data_folder + "/" + dataset1 + "/" + dataset2, file


class SyntheticDataset(Dataset):

    def __init__(self, fpath, params):
        self.x = torch.from_numpy(np.load(fpath + "/x_train.npy"))
        #self.x = torch.from_numpy(np.load(fpath))
        if len(self.x.size()) == 2:
            self.x = self.x.unsqueeze(0)
        self.x_dot = torch.from_numpy(np.load(fpath + "/x_dot.npy"))
        #self.x_dot = self.fourth_order_diff(self.x, hyperparams.dt)
        self.x_lib = sindy_library(
            self.x, params.poly_order,
            include_constant=params.include_constant,
            include_sine=params.include_sine)

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
