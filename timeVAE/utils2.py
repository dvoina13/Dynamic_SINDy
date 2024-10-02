import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import sys
import tensorflow as tf
from config import config as cfg
import derivative
from scipy.special import binom

from tensorflow.keras.layers import Reshape

TITLE_FONT_SIZE = 16

def get_my_data(dataset):
    #folder = "/gscratch/dynamicsai/doris/HyperSINDy/data/sinusoid/" + dataset
    #input_file = "/state-A_B_allIC_harmonic_osc_v100/x_train.npy"
    
    #folder = "/gscratch/dynamicsai/doris/HyperSINDy/data/sinusoid/" + dataset
    #folder = "/home/doris/HyperSINDy/data/sinusoid/" + dataset
    #input_file = "/state-B_allIC_harmonic_osc_short/x_train.npy" #v2 for sharp sigmoid
    #input_file2 = "/state-B_allIC_harmonic_osc_short/x_dot.npy" #v2 for sharp sigmoid

    #input_file = "/state-A_allIC_harmonic_osc_1sinusoid_0419/x_train.npy"
    #input_file2 = "/state-A_allIC_harmonic_osc_1sinusoid_0419/x_dot.npy"
    #input_file = "/state-A_B_allIC_harmonic_osc_2sinusoids_0411/x_train.npy"
    #input_file = "/state-A_allIC_harmonic_osc_0327/x_train.npy"
    #input_file = "/state-s_allIC_lorenz_0327/x_train.npy"
    
    #folder = "/home/doris/HyperSINDy/data/sigmoid/" + dataset
    #input_file = "/state-A_allIC_harmonic_osc_sigmoid_diff0.01_10000/x_train.npy"#"/state-A_allIC_harmonic_osc_sigmoid_std1_0421_ics_rand1/x_train.npy" #v2 or ics for sharp sigmoid, ct 2
    #input_file2 = "/state-A_allIC_harmonic_osc_sigmoid_diff0.01_10000/x_dot.npy"#"/state-A_allIC_harmonic_osc_sigmoid_std1_0421_ics_rand1/x_dot.npy"

    #folder = "/home/doris/HyperSINDy/data/switch_signal/" + dataset
    #input_file = "/state-A_allIC_harmonic_osc_switch_signal_diff0.01/x_train.npy" #v2 or ics for sharp sigmoid, ct 2
    #input_file2 = "/state-A_allIC_harmonic_osc_switch_signal_diff0.01/x_dot.npy"

    #folder = "/home/doris/HyperSINDy/data/random_switches/" + dataset
    #input_file = "/state-A_allIC_harmonic_osc_random_switches_diff0.01/x_train.npy" 
    #input_file2 = "/state-A_allIC_harmonic_osc_random_switches_diff0.01/x_dot.npy"

    #folder = "/home/doris/HyperSINDy/data/Fourier_series/" + dataset
    #input_file = "/state-A_allIC_harmonic_osc_diff_1.0_freq1/x_train.npy"
    #input_file2 = "/state-A_allIC_harmonic_osc_diff_1.0_freq1/x_dot.npy"
    
    #folder = "/home/doris/HyperSINDy/data/drift_diffusion/" + dataset
    #input_file = "/state-A_allIC_harmonic_osc_v0/x_train.npy"
    #input_file2 = "/state-A_allIC_harmonic_osc_v0/x_dot.npy"
    
    #folder = "/home/doris/HyperSINDy/data/sigmoid/" + dataset
    #input_file = "/state-A_allIC_harmonic_osc_diff_0.01/x_train.npy" #v2 or ics for sharp sigmoid, ct 2
    #input_file2 = "/state-A_allIC_harmonic_osc_diff_0.01/x_dot.npy"

    """
    folder = "/home/doris/HyperSINDy/data/Celegans_data/"
    input_file = "2pca_neural_act_worm4.npy"
    input_file2 = "2pca_neural_act_worm4.npy"
    """
    
    #folder = "/home/doris/HyperSINDy/data/sigmoid/" + dataset
    #input_file = "/state-s_allIC_lorenz_sigmoid_diff_10.0/x_train.npy"#"/state-s_allIC_lorenz_v1/x_train.npy" #"/state-s_allIC_lorenz_0327/x_train.npy"
    #input_file2 = "/state-s_allIC_lorenz_sigmoid_diff_10.0/x_dot.npy"#"/state-s_allIC_lorenz_v1/x_dot.npy" #"/state-s_allIC_lorenz_0327/x_dot.npy"
    
    folder = "/home/doris/HyperSINDy/data/sigmoid/" + dataset
    input_file = "/state-A_allIC_harmonic_osc_sigmoid_new1_diff_0.01/x_train.npy"#"/state-alpha_allIC_lotka_volterra_2d_simple/x_train.npy" #"/state-s_allIC_lorenz_simple/x_train.npy" #"lotka_volterra_2d_short/x_train.npy" #"/state-s_allIC_diverse_lorenz_sharp_sigmoid_diff0.1_1000/x_train.npy" #"/state-C_allIC_rossler_diff_0.01/x_train.npy" #"/state-s_allIC_lorenz_sharp_switch_signal_diff0.1_1000/x_train.npy" #(4000) #"/state-s_allIC_lorenz_0327/x_train.npy"#"/state-s_allIC_lorenz_v1/x_train.npy" #"/state-s_allIC_lorenz_diff_0.01_freq/x_train.npy"#"/state-A_allIC_harmonic_osc_diff_0.01_freq_10000/x_train.npy"#"/state-s_allIC_lorenz_sinusoid_diff1.0_1000/x_train.npy" #"/state-s_allIC_lorenz_0327/x_train.npy"
    input_file2 = "/state-A_allIC_harmonic_osc_sigmoid_new1_diff_0.01/x_dot.npy" #/state-alpha_allIC_lotka_volterra_2d_simple/x_dot.npy" #"/state-s_allIC_lorenz_simple/x_dot.npy"  #"lotka_volterra_2d_short/x_dot.npy" #"/state-C_allIC_rossler_diff_0.01/x_dot.npy" #"/state-s_allIC_lorenz_sharp_switch_signal_diff0.1_1000/x_dot.npy" #"/state-s_allIC_lorenz_0327/x_train.npy"#"/state-s_allIC_lorenz_v1/x_dot.npy" #"/state-s_allIC_lorenz_diff_0.01_freq/x_dot.npy"#"/state-A_allIC_harmonic_osc_diff_0.01_freq_10000/x_dot.npy"#"/state-s_allIC_lorenz_sinusoid_diff1.0_1000/x_dot.npy"#"/state-A_allIC_harmonic_osc_sharp_sigmoid_diff0.01_6000/x_dot.npy"#

    print("input_file", input_file)
    data = np.load(folder + input_file)
    data_dot = np.load(folder + input_file2)

    #return np.expand_dims(data[0,:,:],0), np.expand_dims(data_dot[0,:,:],0)
    return data, data_dot

def get_multiple_trajectories(x, len_seq=700):
    T,dim = x.shape
    print("T, dim", T, dim) 
        
    if T!=len_seq:
        new_x = np.array([x[i:i+len_seq,:] for i in range(T-len_seq)])
        idx_perm = np.random.permutation(T-len_seq)
        new_x = new_x[idx_perm,:,:]

    else:
        new_x = x.reshape(1,T,dim)  
    #new_x = np.array([x[i:i+len_seq,:] for i in range(300)])
    
    return new_x

def get_training_data(input_file):
    loaded = np.load(input_file)
    return loaded['data']



def get_daily_data():
    data = pd.read_parquet(cfg.DATA_FILE_PATH_AND_NAME)
    data.rename(columns={ 'queueid': 'seriesid', 'date': 'ts', 'callvolume': 'v',}, inplace=True)
    data['ts'] = pd.to_datetime(data['ts'])
    data = data[['seriesid', 'ts', 'v']]
    return data


def get_mnist_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    # mnist_digits = np.concatenate([x_train, x_test], axis=0)
    # mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_digits = x_train.astype("float32") / 255
    return mnist_digits
    
def draw_orig_and_post_pred_sample(orig, reconst, n):

    fig, axs = plt.subplots(n, 2, figsize=(10,6))
    i = 1
    for _ in range(n):
        rnd_idx = np.random.choice(len(orig))
        o = orig[rnd_idx]
        r = reconst[rnd_idx]

        plt.subplot(n, 2, i)
        plt.imshow(o, 
            # cmap='gray', 
            aspect='auto')
        # plt.title("Original")
        i += 1

        plt.subplot(n, 2, i)
        plt.imshow(r, 
            # cmap='gray', 
            aspect='auto')
        # plt.title("Sampled")
        i += 1

    fig.suptitle("Original vs Reconstructed Data", fontsize = TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.show()


def plot_samples(samples, n):    
    fig, axs = plt.subplots(n, 1, figsize=(6,8))
    i = 0
    for _ in range(n):
        rnd_idx = np.random.choice(len(samples))
        s = samples[rnd_idx]
        axs[i].plot(s)    
        i += 1

    fig.suptitle("Generated Samples (Scaled)", fontsize = TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.show()


def plot_latent_space_timeseries(vae, n, figsize):
    scale = 3.0
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    grid_size = len(grid_x)

    Z2 = [ [x, y]  for x in grid_x for y in grid_y ]
    X_recon = vae.get_prior_samples_given_Z(Z2)
    X_recon = np.squeeze(X_recon)
    # print('latent space X shape:', X_recon.shape)

    
    fig, axs = plt.subplots(grid_size, grid_size, figsize=figsize)
    k = 0
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            x_recon = X_recon[k]
            k += 1            
            axs[i,j].plot(x_recon)
            axs[i,j].set_title(f'z1={np.round(xi, 2)};  z2={np.round(yi,2)}')
    
    
    fig.suptitle("Generated Samples From 2D Embedded Space", fontsize = TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.show()



def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    Z2 = [ [x, y]  for x in grid_x for y in grid_y ]
    X_recon = vae.get_prior_samples_given_Z(Z2)
    X_recon = np.squeeze(X_recon)
    # print(X_recon.shape)
    
    k = 0
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            x_decoded = X_recon[k]
            k += 1
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = x_decoded

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()



# Custom scaler for 3d data
class MinMaxScaler_Feat_Dim():
    '''Scales history and forecast parts of time-series based on history data'''
    def __init__(self, scaling_len, input_dim, upper_bound = 3., lower_bound = -3.):         
        self.scaling_len = scaling_len
        self.min_vals_per_d = None      
        self.max_vals_per_d = None  
        self.input_dim = input_dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        

    def fit(self, X, y=None): 

        if self.scaling_len < 1: 
            msg = f''' Error scaling series. 
            scaling_len needs to be at least 2. Given length is {self.scaling_len}.  '''
            raise Exception(msg)

        X_f = X[ :,  : self.scaling_len , : ]
        self.min_vals_per_d = np.expand_dims(np.expand_dims(X_f.min(axis=0).min(axis=0), axis=0), axis=0)
        self.max_vals_per_d = np.expand_dims(np.expand_dims(X_f.max(axis=0).max(axis=0), axis=0), axis=0)

        self.range_per_d = self.max_vals_per_d - self.min_vals_per_d
        self.range_per_d = np.where(self.range_per_d == 0, 1e-5, self.range_per_d)

        # print(self.min_vals_per_d.shape); print(self.max_vals_per_d.shape)
              
        return self
    
    def transform(self, X, y=None): 
        assert X.shape[-1] == self.min_vals_per_d.shape[-1], "Error: Dimension of array to scale doesn't match fitted array."
         
        X = X - self.min_vals_per_d
        X = np.divide(X, self.range_per_d )        
        X = np.where( X < self.upper_bound, X, self.upper_bound)
        X = np.where( X > self.lower_bound, X, self.lower_bound)
        return X
    
    def fit_transform(self, X, y=None):
        X = X.copy()
        self.fit(X)
        return self.transform(X)
        

    def inverse_transform(self, X):
        X = X.copy()
        X = X * self.range_per_d 
        X = X + self.min_vals_per_d
        # print(X.shape)
        return X



class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        
        print("self.mini, self.range", self.mini, self.range)
        return self
        

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data



def library_size(n, poly_order, use_sine=False, use_mult_sine=False, include_constant=False):
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

def sindy_library(X, poly_order=3, include_sine=False, include_mult_sine=False, include_constant=False, device="cpu"):
    # batch x latent dim
    #X = np.array(X)
    #print("X.shape", X.shape)
    if len(X.shape)>3:
        X = Reshape(target_shape=(-1, X.shape()[2], X.shape()[3]))(X)
    
    b, m, n = X.shape
    print("b,m,n", b,m,n)
    l = library_size(n, poly_order)
    print("l", l)
    #library = tf.fill([b,m,l], 0.0)
    #library = tf.constant(1.0, dtype=tf.float32, shape = [b, m,l])
    library = []
    
    index = 0

    if include_constant:
        library.append(tf.where(tf.math.is_nan(X[:,:,index]/X[:,:,index]), tf.ones_like(X[:,:,index]), X[:,:,index]/X[:,:,index]))
        index = 1

    for i in range(n):
        library.append(X[:, :, i])
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i, n):
                library.append(X[:, :, i] * X[:, :, j])
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    library.append(X[:, :, i] * X[:, :, j] * X[:, :, k])
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        library.append(X[:, :, i] * X[:, :, j] * X[:, :, k] * X[:, :, q])
                        index += 1

    if poly_order > 4:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        for r in range(q, n):
                            library.append(X[:, :, i] * X[:, :, j] * X[:, :, k] * X[:, :, q] * X[:, :, r])
                            index += 1

    if include_sine:
        for i in range(n):
            library.append(torch.sin(X[:, :, i]))
            index += 1

    if include_mult_sine:
        for i in range(n):
            library.append(torch.sin(X[:, :, i]))
            index += 1

    library = tf.transpose(tf.stack(library), [1, 2, 0])
    #print("lb shape", library.shape)
    #library = library.tolist()
    #library = [np.array(library[i]) for i in range(len(library))]
    return library



def sindy_library2(X, poly_order=3, include_sine=False, include_mult_sine=False, include_constant=False, device="cpu"):
    # batch x latent dim
    #X = np.array(X)
    #print("X.shape", X.shape)
    if len(X.shape)>3:
        X = Reshape(target_shape=(-1, X.shape()[2], X.shape()[3]))(X)
    
    b, m, n = X.shape
    print("b,m,n", b,m,n)
    l = library_size(n, poly_order)
    print("l", l)
  
    library = []
    

    library.append(X[:, :, 0])
    library.append(X[:, :, 1])
    library.append(X[:, :, 0])
    library.append(X[:, :, 1])
    library.append(X[:, :, 0]*X[:, :, 2])
    library.append(X[:, :, 2])
    library.append(X[:, :, 0]*X[:, :, 1])
    
    library = tf.transpose(tf.stack(library), [1, 2, 0])

    return library


def sindy_library_rossler(X, poly_order=3, include_sine=False, include_mult_sine=False, include_constant=False, device="cpu"):
    # batch x latent dim
    #X = np.array(X)
    #print("X.shape", X.shape)
    if len(X.shape)>3:
        X = Reshape(target_shape=(-1, X.shape()[2], X.shape()[3]))(X)
    
    b, m, n = X.shape
    print("b,m,n", b,m,n)
    l = library_size(n, poly_order)
    print("l", l)
  
    library = []
    

    library.append(X[:, :, 1])
    library.append(X[:, :, 2])
    library.append(X[:, :, 0])
    library.append(X[:, :, 1])
    library.append(tf.ones_like(X[:, :, 0]))
    library.append(X[:, :, 2])
    library.append(X[:, :, 0]*X[:, :, 2])
    
    library = tf.transpose(tf.stack(library), [1, 2, 0])

    return library

"""
def normalize_SINDy(library):
    
    scaling = np.array([np.linalg.norm(library[i,:,:], axis=1) for i in range(library.shape[0])])
    print("library shape", library.shape)
    print("scaling shape", scaling.shape)
    print("diag shape", np.diag(scaling).shape, np.diag(scaling))
    normalized_library = library#np.diag(scaling)
    print("normalized_library", normalized_library.shape)
    return normalized_library, scaling
"""

def normalize_SINDy(library):

    print("library size", library.shape)
    scaling = 1#np.array([np.diag(1/np.linalg.norm(library[i,:,:], axis=0)) for i in range(library.shape[0])])
    #print("scaling size", scaling.shape)
    normalized_library = library#np.matmul(library,scaling) #library
    return normalized_library, scaling

def compute_derivative(data, T, dt):
        N, T, D = data.shape
        
        differentiator = derivative.Kalman(alpha=0.5)

        data = np.transpose(data,(0,2,1)).reshape(-1,T)
        print("data.shapeee", data.shape)
        #data_dot = np.transpose( differentiator.d(data, np.arange(0,dt*T, dt)[:-1]) )
        data_dot = np.transpose( differentiator.d(data, np.arange(0,dt*T, dt)).reshape(N,D,T), (0,2,1) )  
        print("data_dot.shapeee", data_dot.shape)
        
        return data_dot

if __name__ == '__main__':

    # data = get_daily_data()
    data = get_my_data()
    print(data.shape)