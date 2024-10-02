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
import pickle

from args_params_hyperparams import parse_hyperparams, parse_args
from load_data import load_data, str_filename, get_sub_data_path, get_data_path, integrate_datasets, MultipleSyntheticDatasets, SyntheticDataset
from utils1 import library_size, sindy_library, init_weights
from utils2 import update_threshold_mask, update_beta

from models_rnn_for_mixBasis import HyperNet, rHyperNet, Net
from regularization import _L0Norm, hard_sigmoid, smoothness_reg

def make_folder(name):
    if not os.path.isdir(name):
        os.makedirs(name)

args = parse_args()
hyperparams = parse_hyperparams()

if torch.cuda.is_available():
    # device = torch.cuda.device(3) #torch.device('cuda')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# seed = args.seed

# torch.manual_seed(seed)
# if device.type == "cuda":
#    torch.cuda.manual_seed(seed)

# np.random.seed(seed)

train_set = load_data(hyperparams, args)

# this worked OK
train_set.x = train_set.x[:100, :1000, :]
train_set.x_dot = train_set.x_dot[:100, :1000, :]
train_set.x_lib = train_set.x_lib[:100, :1000, :]

trainloader = DataLoader(train_set, batch_size=hyperparams.batch_size,
                         shuffle=False, drop_last=True)

net = Net(args, hyperparams).to(device)
net.apply(init_weights)

if args.saved_model:
    print("I'm loading model")
    path = args.path_model + "/model_epoch" + str(19) + "_trial_" + str(9) + ".pt"
    net.load_state_dict(torch.load(path))
else:
    print("I'm not loading model")

# optim = torch.optim.Adam(
#        net.parameters(), lr=hyperparams["learning_rate"],
#        weight_d   ecay=hyperparams["adam_reg"], amsgrad = hyperparams["amsgrad"])

optim = torch.optim.AdamW(
    net.parameters(), lr=hyperparams.learning_rate,
    weight_decay=hyperparams.adam_reg,
    amsgrad=hyperparams.amsgrad)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optim, gamma=args.gamma_factor)

weight_decay_l0 = hyperparams.weight_decay_l0  #####
use_l0 = hyperparams.use_l0  #####

# net = net.train()
epochs = hyperparams.epochs
beta = hyperparams.beta_init
if hyperparams.beta_inc is None:
    hyperparams.beta_inc = hyperparams.beta / 5

# train_interval = hyperparams.train_interval
smooth_interval = hyperparams.smooth_interval

recon_smoothed = None

for epoch in range(20):
    Sindy_coeffs22 = []

    print("!! EPOCH", epoch)
    net.train()
    recons, klds = 0, 0

    for ii, (x, x_lib, x_dot) in enumerate(trainloader):

        # Loss = []
        # Recon = []
        # KLDs = []

        Sindy_coeffs_int22 = torch.zeros(10000, 19, 3)
        Sindy_coeffs_test = torch.zeros(10000, 19, 3)
        x_dot_pred_saved = []

        # torch.cuda.empty_cache()
        print("iii", ii)
        j = 0

        x = x.permute(1, 0, 2);
        x_dot = x_dot.permute(1, 0, 2);
        x_lib = x_lib.permute(1, 0, 2);
        x_saved = x.detach().clone();
        x_dot_saved = x_dot.detach().clone();
        x_lib_saved = x_lib.detach().clone();

        train_data_new = MultipleSyntheticDatasets(x, x_dot, x_lib)
        train_data_new_loader = DataLoader(train_data_new, batch_size=hyperparams.len_seq,
                                           shuffle=False, drop_last=True)

        for i, (x, x_lib, x_dot) in enumerate(train_data_new_loader):
            print("epoch", epoch)
            print("i", i)
            print("len seq", hyperparams.len_seq)

            j += hyperparams.batch_size

            x = x.permute(1, 0, 2);
            x_dot = x_dot.permute(1, 0, 2);
            x_lib = x_lib.permute(1, 0, 2);
            # print("some sizes4", x.size(), x_lib.size(), x_dot.size())

            x_dot = x_dot.type(torch.FloatTensor).to(device)
            if i == 0:
                z = None;
                c = None;
                step = 0;
            else:
                z = out_level2.detach().clone().to(device);
                step = i * hyperparams.len_seq

                if hyperparams.lstm:
                    c = out_level2_c.detach().clone().to(device);
                else:
                    c = None

            x_dot_pred, sindy_coeffs, out_level2, out_level2_c, input_to_hypernet, reg, mu, logvar = net(x.to(device), x_dot.to(device), z, c, step, None, device)
            x_dot_pred_saved.append(x_dot_pred)

            Sindy_coeffs_int22[i * hyperparams.len_seq:(i + 1) * hyperparams.len_seq, :, :] = net.sindy_coefficients

            recon = ((x_dot_pred - x_dot) ** 2).sum(2).mean()
            kld = torch.abs(sindy_coeffs).mean() #((sindy_coeffs)**2).mean()
            # kld = net.kl(mu, logvar)
            smooth = smoothness_reg(sindy_coeffs)

            print("recon, kld, smooth", recon, kld, smooth)
            print("sindy_coefficients", sindy_coeffs.size(), sindy_coeffs[0, 0, :, :])

            """
            reg1 = 0
            for j in range(len(input_to_hypernet)):
                reg1 += torch.abs(input_to_hypernet[j]).sum()
            reg2 = ((net.sindy_coefficients) ** 2).mean()
            #print("recon, reg1, kld", recon, reg1 * hyperparams["beta_reg2"], reg2 * hyperparams["beta_reg3"], kld)
            """

            if use_l0:  #####
                # loss = recon + weight_decay_l0 * reg #####
                # loss = recon + weight_decay_l0 * reg + kld * beta + smooth # + torch.abs(Weights).sum() * hyperparams["beta_reg2"]  + reg2 *hyperparams["beta_reg3"] #+ kld * beta
                loss = recon + weight_decay_l0 * reg + smooth #+ kld * beta#+ reg2 * hyperparams.beta3  # + torch.abs(Weights).sum() * hyperparams["beta_reg2"]  + reg2 *hyperparams["beta_reg3"] #+ kld * beta
                # print("recon, reg, reg2", recon, reg, reg2, reg2 * hyperparams.beta3)
            else:
                loss = recon
                # + kld * beta #+ torch.abs(Weights).sum() * hyperparams["beta_reg2"]  + reg2 *hyperparams["beta_reg3"] #+ kld * beta
            # print("size of net.sindy_coefficients", net.sindy_coefficients.size())

            """
            if recon_smoothed != None:
                print("!!!*(*E@# recon smooth 10000", recon_smoothed.size(), loss, recon_smoothed)
                print("loss before", loss)
                #loss += recon_smoothed
                print("new loss now! 0-:", loss)
            """

            # Loss.append(loss.item())
            # Recon.append(recon.item())
            # KLDs.append(kld)

            # if epoch > 10:
            #    decide_requires_grad(i, train_interval)
            # for n, p in net.named_parameters():
            #    print(i, n, p.requires_grad)

            optim.zero_grad()
            loss.backward(retain_graph=True)
            # loss.backward()
            if hyperparams.clip is not None:
                nn.utils.clip_grad_norm_(net.parameters(), hyperparams.clip)

            optim.step()

            recons += recon
            # klds += kld

        # print("len Sindy_coeffs_int22", len(Sindy_coeffs_int22), Sindy_coeffs_int22[0].size())

        Sindy_coeffs_int22 = Sindy_coeffs_int22.squeeze().unsqueeze(0)
        Sindy_coeffs22 = Sindy_coeffs_int22.reshape(10000, 19, 3)

        # if (epoch % 50 == 0) and (trial == 0) and hyperparams.sparsify_derivative:
        #    net.sparsify_sindy_dot(Sindy_coeffs_int22)

        #folder = "/home/doris/HyperSINDy/hello/saved_results/new_hyak/lorenz_sigmoid_len_seq" + str(
        folder = hyperparams.experiments + "/" + hyperparams.dataset1 + "_" + hyperparams.dataset2 + "_len_seq" + str(
            hyperparams.len_seq) + "/Sindy_coeffs_" + hyperparams.dataset1 + "_" + hyperparams.dataset2 + "_T" + str(hyperparams.threshold) + "_Tincr" + str(
            hyperparams.threshold_increment) + "_TI" + str(hyperparams.threshold_interval) + "_" + hyperparams.param[0] + "_lenseq_" + str(
            hyperparams.len_seq) + "_long_no_ae_10trials_SimpleHyperSINDy_no_kld"  # CHANGE

        make_folder(folder)

        torch.save(Sindy_coeffs22, folder + "/Sindy_coeffs_epoch" + str(epoch) + "_trial_" + str(ii))
        torch.save(net.state_dict(), folder + "/model_epoch" + str(epoch) + "_trial_" + str(ii) + ".pt")

        sindy_coeffs_all = Sindy_coeffs22.unsqueeze(3)
        del Sindy_coeffs_int22, Sindy_coeffs22, reg

        # np.save(folder + "/Recon_epoch" + str(epoch) + "_trial_" + str(ii), np.array(Recon))
        # np.save(folder + "/KLD_epoch" + str(epoch) + "_trial_" + str(ii), torch.stack(KLDs).cpu().detach().numpy())
        # del Recon, KLDs
        ####smooth out trajectory and learn

        """
        net.to("cpu")

        optim2 = torch.optim.AdamW(
            net.parameters(), lr=hyperparams.learning_rate,
            weight_decay=hyperparams.adam_reg,
            amsgrad=hyperparams.amsgrad)

        _, sindy_coeffs_all, _, _, _, _ = net(x_saved, None, None, 0, x_lib_saved.unsqueeze(0), device="cpu")
        Sindy_coeffs_test = sindy_coeffs_all

        ========


        Sindy_coeffs_smoothed = Variable(smooth_function(sindy_coeffs_all.detach(), 1000), requires_grad=True)
        x_dot_predicted = net.dx(x_lib_saved.unsqueeze(0), Sindy_coeffs_smoothed)
        torch.save(Sindy_coeffs_smoothed, folder + "/Sindy_coeffs_smoothed_epoch" + str(epoch) + "_trial_" + str(ii))
        del x_saved, x_lib_saved, Sindy_coeffs_test

        recon_smoothed = Variable(((x_dot_predicted.detach().squeeze() - x_dot_saved.squeeze()) ** 2).sum(1), requires_grad=True)
        #recon_smoothed = Variable(((Sindy_coeffs_smoothed.detach().squeeze() - sindy_coeffs_all.detach().squeeze()) ** 2).view(10000, -1).sum(1), requires_grad=True)

        del x_dot_predicted, x_dot_saved


        ======

        loss = recon

        optim2.zero_grad()
        loss.backward(retain_graph=True)
        # loss.backward()
        if hyperparams.clip is not None:
            nn.utils.clip_grad_norm_(net.parameters(), hyperparams.clip)

        optim2.step()
        net.to(device)
        """
        ####

    hard_threshold = update_threshold_mask(net, hyperparams.threshold, hyperparams.threshold_increment,
                                           hyperparams.threshold_interval, epoch, device,
                                           beta, hyperparams.beta, folder)
    scheduler.step()
    beta = update_beta(beta, hyperparams.beta_inc, hyperparams.beta)

with open('hyperparams.pkl', 'wb') as fp:
        pickle.dump(hyperparams, fp)

with open('args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

# eval_model(net.eval(), params, train_set, device, epoch)

