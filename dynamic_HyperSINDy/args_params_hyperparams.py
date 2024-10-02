import argparse


# These are hyperparameters that get logged into tensorboard

def parse_hyperparams():
    parser = argparse.ArgumentParser(description="Template")

    #parser.add_argument('-DF', '--data_folder', default='/gscratch/dynamicsai/doris/HyperSINDy/data/', type=str,
    #                    help="Base folder where all data is stored")
    parser.add_argument('-DF', '--data_folder', default='/home/doris/HyperSINDy/data', type=str,
                            help="Base folder where all data is stored")
    parser.add_argument('-DS1', '--dataset1', default="sigmoid", type=str, help="Which dataset to use (sigmoid)")
    parser.add_argument('-DS2', '--dataset2', default="lorenz", type=str, help="Which dataset to use (lorenz)")
    parser.add_argument('-DFI', '--data_file', default="state-s_allIC_lorenz_long_v2")
    parser.add_argument('-EX', '--experiments', default='/data/doris/', type=str,
                        help="Output folder for experiments")

    parser.add_argument('-DT', '--dt', default=0.01, type=float, help='Time change in dataset')

    parser.add_argument('-ND1', '--noise_dim1', default=25, type=int, help="Noise vector dimension for HyperSINDy")
    parser.add_argument('-ND2', '--noise_dim2', default=25, type=int, help="Noise vector dimension for HyperSINDy")
    parser.add_argument('-HD1', '--hidden_dim1', default=256, type=str, help="Dimension of hidden layers hypernet")
    parser.add_argument('-HD2', '--hidden_dim2', default=256, type=str, help="Dimension of hidden layers hypernet")

    parser.add_argument('-BS', '--batch_size', default=1, type=int, help="Batch size")
    parser.add_argument('-LENSQ', '--len_seq', default=1, type=int, help="Sequence length in RNN")
    parser.add_argument('-CELLD', '--cell_dim', default=30, type=int, help="Cell dimension for LSTM")

    parser.add_argument('-NLAYERS', '--num_layers_rnn', default=2, type=int, help="Number of layers for RNN")

    parser.add_argument('-LR', '--learning_rate', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('-AR', '--adam_reg', default=1e-5, type=float, help="Regularization to use in ADAM optimizer")

    # Primarily for SINDy / ESINDy, but works as an extra regularization term for HyperSINDy (recommended None for HyperSINDy)
    parser.add_argument('-WD', '--weight_decay', default=None, type=float,
                        help="Weight decay for sindy coefficients (None to disable)")
    parser.add_argument('-C', '--clip', default=1.0, type=float,
                        help="Gradient clipping value during training (None for no clipping)")

    parser.add_argument('-USEL0', '--use_l0', default=True, type=bool, help="Whether to use L0 norm as regularization for (Hyper)SINDy coefficients")
    parser.add_argument('-WD_L0', '--weight_decay_l0', default=1e-5, type=float, help="weight for L0 regularization")

    parser.add_argument('-B', '--beta', default=1.0, type=float,
                        help="KL divergence weight in loss (only for HyperSINDy)")
    parser.add_argument('-BINIT', '--beta_init', default=0.01, type=float, help="Inital beta value")
    parser.add_argument('-BINCR', '--beta_inc', default=None, type=float,
                        help="Beta increment per epoch till beta max. If none, = beta_max / 100")
    parser.add_argument('-B2', '--beta2', default=100.0, type=float,
                        help="smoothness regularization weight in loss (only for HyperSINDy)")
    parser.add_argument('-B3', '--beta3', default=0.0, type=float,
                        help="other regularization weight in loss (only for HyperSINDy)")

    parser.add_argument('-AMS', '--amsgrad', default=True, type=bool, help="IFF True, uses amsgrad in Adam optimizier.")

    parser.add_argument('-E', '--epochs', default=100, type=float, help="Number of epochs to train for")
    parser.add_argument('-RND', '--random', default=True, type=bool,
                        help='have the input vector to the hyper net be a random vector or a deterministic vector (of ones/normalization)')

    parser.add_argument('-ST', '--soft_threshold', default=0.1, type=float,
                        help="Soft threshold to 0 coefficient samples. 0 to disable.")
    parser.add_argument('-SS', '--sigmoid_scale', default=1.0, type=float,
                        help="Scaling factor inside soft threshold sigmoid.")
    parser.add_argument('-LTSS', '--learn_threshold_and_sigmoid', default=True, type=float,
                        help="Learn soft threshold and sigmoid scale")

    parser.add_argument('-HT', '--threshold', default=0.1, type=float,
                        help="Hard threshold to permanently 0 coefficients out. Updated every threshold_interval epochs. 0 to disable.")
    parser.add_argument('-TIN', '--threshold_increment', default=0.005, type=float,
                        help="Every threshold_interval, increases the hard threshold by this amount.")
    parser.add_argument('-TI', '--threshold_interval', default=5, type=float,
                        help="Epoch interval in training to permanently threshold sindy coefs")

    parser.add_argument('-PARAM', '--param', default=["s"], type=list, help="parameter that varies in the dataset")
    parser.add_argument('-DRIFT', '--drift', default=[1.0], type=list, help="drift of parameter that varies in the dataset")
    parser.add_argument('-DIFF', '--diff', default=[1.0], type=list, help="diffusion of parameter that varies in the dataset")
    parser.add_argument('-USE_IC', '--use_all_ic', default=True, type=bool, help="use all initial conditions (IC) in the dataset")
    parser.add_argument('-IC', '--ic', default=1, type=float, help="Initial condition to use in the dataset")

    parser.add_argument('-NOISET', '--noise_type', default='x', type=str,
                        help='Type of state-dependent noise (x, sinz)')
    parser.add_argument('-NOISES', '--noise_scale', default=1.0, type=float,
                        help='Scale of noise in data. Review data folder.')

    parser.add_argument('-P', '--prior', default="normal", type=str, help="Prior to regularize to. Options: laplace, normal. For SINDy, uses L1 or L2 norm, respectively. For HyperSINDy, affects KL.")

    parser.add_argument('-TR', '--train_interval', default=30, type=int, help="how many iterations to train either RNN/LSTM versus hypernetwork before switching")
    parser.add_argument('-SM', '--smooth_interval', default=10, type=int, help="how many timesteps to use for smoothing")

    parser.add_argument('-MC', '--model_choice', default = "mixed basis for hypernet", type=str, help="What type of model to use. There are two options: (1) input of RNN/LSTM used directly (2) input to hypernet; (3) mixed basis for hypernet")
    parser.add_argument('-LSTM', '--lstm', default=True, type=bool, help="LSTM or RNN? whether to use LSTM or not")

    parser.add_argument('-AE', '--autoencoder', default=False, type=bool, help="Apply autoencoder")

    parser.add_argument('-SPRS', '--sparsify_derivative', default=False, type=int, help="Should I apply the sparsification of the derivative of sindy coefficients?")
    parser.add_argument('-HNET', '--hypernets', default=1, type=int, help="Number of hypernets to use")

    return parser.parse_args()


def parse_args():
    parser = argparse.ArgumentParser(description="Template")

    parser.add_argument('-SEED', '--seed', default=0, type=int,
                        help="Define seed")
    parser.add_argument('-MF', '--model_folder', default='./trained_models/', type=str,
                        help="Output folder for experiments")
    parser.add_argument('-TB', '--tensorboard_folder', default='./runs/experiments/', type=str,
                        help="Output folder for tensorboard")

    # saving specifics
    parser.add_argument('-sess', '--session_name', default='222', type=str, help="Appended to last part of file names")
    parser.add_argument('-DAT', '--date', default="01-23-23", type=str, help="The date"),
    parser.add_argument('-M', '--model', default="HyperSINDy", type=str, help="Model to use")
    parser.add_argument('-MS', '--saved_model', default=False, type=bool, help="Model to load from saved and learned path")
    parser.add_argument('-PM', '--path_model', default="/data/doris/lorenz_sinusoid_len_seq1__rnn_layers2/Sindy_coeffs_lorenz_sinusoid_sigma_AE_lenseq_1_2random_1_noFC_fixedHT", type=str, help="Path to saved model")
    # sindy parameters
    parser.add_argument('-Z', '--z_dim', default=3, type=int, help="Size of latent vector")
    parser.add_argument('-PO', '--poly_order', default=3, type=int, help="Size of theta library for SINDy")
    parser.add_argument('-INCC', '--include_constant', default=False, type=bool,
                        help="IFF True, includes sine in SINDy library")
    parser.add_argument('-INCS', '--include_sine', default=False, type=bool,
                        help="IFF True, includes sine in SINDy library")

    # training parameters
    parser.add_argument('-GF', '--gamma_factor', default=0.999, type=float, help="Learning rate decay factor")
    parser.add_argument('-CPI', '--checkpoint_interval', default=25, type=float,
                        help="Epoch interval to save model during training")

    # dataset parameters
    parser.add_argument('-ND', '--norm_data', default=False, type=bool, help='Iff true, normalizes data to N(0, 1)')
    parser.add_argument('-SD', '--scale_data', default=0.0, type=int,
                        help='Scales the data values (after normalizing).')

    # experiment parameters
    parser.add_argument('-EBS', '--exp_batch_size', default=10, type=int, help='Batch size for experiments')
    parser.add_argument('-ETS', '--exp_timesteps', default=100, type=int, help='Number of timesteps per trajectory')

    parser.add_argument('-EI', '--eval_interval', default=1, type=float,
                        help="Epoch interval to evalate model during training")

    # other
    parser.add_argument('-D', '--device', default=1, type=int, help='Which GPU to use')
    parser.add_argument('-LCP', '--load_cp', default=0, type=int,
                        help='If 1, loads the model from the checkpoint. If 0, does not')
    parser.add_argument('-PF', '--print_folder', default=1, type=int,
                        help='Iff true, prints the folder for different logs')
    parser.add_argument('-SBS', '--statistic_batch_size', default=500, type=str, help="Default batch size to sample")

    parser.add_argument('-MXD', '--mix_dim', default=10, type = int, help="Number of bases to take linear combination of")
    
    return parser.parse_args()