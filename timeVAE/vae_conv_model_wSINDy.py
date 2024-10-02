import os, warnings, sys
from re import T
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np
import math

import tensorflow as tf
from tensorflow.keras.layers import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Permute, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.backend import random_normal
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop
import tensorflow_probability as tfp
import tensorflow.keras.regularizers as regularizers
import math
import argparse

from utils2 import get_my_data, get_multiple_trajectories, draw_orig_and_post_pred_sample, plot_latent_space, library_size, sindy_library, normalize_SINDy, compute_derivative, plot_latent_space_timeseries, MinMaxScaler_Feat_Dim, MinMaxScaler
from vae_base_wSINDy import BaseVariationalAutoencoder, Sampling
import derivative

class VariationalAutoencoderConv(BaseVariationalAutoencoder):
    def __init__(self, 
            hidden_layer_sizes, 
            **kwargs           
        ):
        super(VariationalAutoencoderConv, self).__init__(**kwargs)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder() 

    def _get_encoder(self):
        #encoder_inputs = Input(shape=(self.seq_len, self.z_dim+10), name='encoder_input')
        encoder_inputs = Input(shape=(self.seq_len, self.z_dim), name='encoder_input')
        x = encoder_inputs
        print("xxx shape", x.shape)
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                    filters = num_filters, 
                    kernel_size=3, 
                    strides=2, 
                    activation='relu', 
                    padding='same',
                    activity_regularizer=regularizers.L2(0.01),
                    name=f'enc_conv_{i}')(x)
            print("xxxx shape", x.shape)
        
        print("xxxx shape", x.shape)
        x = Flatten(name='enc_flatten')(x)
        print("xxxxx shape", x.shape)
        
        # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        self.encoder_last_dense_dim = x.get_shape()[-1]        
         
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        print("z_mean, z_log_var", z_mean, z_log_var)
        
        
        encoder_output = Sampling()([z_mean, z_log_var])     
        self.encoder_output = encoder_output
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder")
        # encoder.summary()
        return encoder

    def _get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim), name='decoder_input')        
        
        x = decoder_inputs
        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation='relu')(x)
        #, activity_regularizer=regularizers.L2(0.01))(x)
        x1 = tf.identity(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(x)
        x2 = tf.identity(x)

        print("x shape", x.shape)        
        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters = num_filters, 
                    kernel_size=3, 
                    strides=2, 
                    padding='same',
                    activation='relu', 
                    activity_regularizer=regularizers.L2(1), #changed for lorenz
                    name=f'dec_deconv_{i}')(x)
            print("x4 decoder shape", x.shape)
            x3 = tf.identity(x)
       
        # last de-convolution
        x = Conv1DTranspose(
                filters = self.feat_dim, 
                    kernel_size=3, 
                    strides=2, 
                    padding='same',
                    activation='relu', 
                    activity_regularizer=regularizers.L2(1), #changed for lorenz
                    name=f'dec_deconv__{i+1}')(x)
        
        x4 = tf.identity(x)
        print("x5 decoder shape", x.shape)
        
        print("s flatten really a problem? decoder")
        x = Flatten(name='dec_flatten')(x)
        print("x6 decoder shape", x.shape)
        #x = Dense(self.seq_len * self.feat_dim, activation="relu", activity_regularizer=regularizers.L2(1), name="decoder_dense_third_final")(x)
        #x = Dense(self.seq_len * self.feat_dim, activation="relu", activity_regularizer=regularizers.L2(1), name="decoder_dense_second_final")(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final",)(x)
        print("x7 decoder shape", x.shape)
        self.decoder_outputs = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        x5 = tf.identity(self.decoder_outputs)
        print("x", self.decoder_outputs)
        print("x8 decoder shape", self.decoder_outputs.shape)
        
        decoder = Model(decoder_inputs, [10*self.decoder_outputs, x1, x2, x3, x4, x5], name="decoder")
        return decoder
    

def add_weight_decay(model):
    
    alpha = model.alpha
    for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(alpha)(layer.kernel))
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(alpha)(layer.bias))
                    
    return model

                  
class MyCallback(tf.keras.callbacks.Callback):
    #def __init__(self, model):
    #    self.model = model
        
    def on_batch_end(self, batch, logs=None):
        
        self.model.batch = batch
        
        print("self.model.epoch and batch", self.model.epoch, batch)
        print("len", len(self.model.masked_sindy_coefficients.shape))
        if (self.model.epoch>=1 and batch%5==0):# or (self.model.epoch==0 and batch>=50 and batch%5==0):
            if len(self.model.masked_sindy_coefficients.shape) == 4:
                print("here1")
                masked_sindy_coeff_reduced = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(self.model.masked_sindy_coefficients), axis=0), axis=0)
                
            self.model.mask = self.model.reconfigure_mask(masked_sindy_coeff_reduced, self.model.epoch)
        
        np.save("makes_sindy_at_batch_end.npy", self.model.masked_sindy_coefficients)
        """
        if batch == 0:
            masked_sindy_coeff_reduced = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(self.model.masked_sindy_coefficients), axis=0), axis=0)
            self.model.mask = self.model.reconfigure_mask(masked_sindy_coeff_reduced, self.model.epoch)
        """
    def on_epoch_begin(self, epoch, logs=None):
        self.model.epoch = epoch
        self.model.scaling = []
        print("begin epoch", self.model.epoch)
        
    def on_epoch_end(self, epoch, logs=None):
        
        #if epoch == 0:
        #    self.model.sparsity = 100;
        #if epoch == 2:
        #    self.model.total_var_coeff = 1000
        
        if epoch >= 6:
            self.model.sparsity = 0
        #if epoch>=1:
        #    #self.threshold = 0.1
        #    if self.model.sparsity < 20:
        #        self.model.sparsity += 1
        #    if self.model.threshold < 0.1:
        #        self.model.threshold += 0.025
        print("I'm at the epoch end") 
        tf.print(tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(self.model.masked_sindy_coefficients), axis=0), axis=0), "hereee, self.masked_sindy_coefficients")
        
        if len(self.model.masked_sindy_coefficients.shape) == 4:
            masked_sindy_coeff_reduced = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(self.model.masked_sindy_coefficients), axis=0), axis=0)
            #print(masked_sindy_coeff_reduced, "masked_sindy_coeff_reduced")
            print("epoch before reconfigure", epoch)
            self.model.mask = self.model.reconfigure_mask(masked_sindy_coeff_reduced, epoch)
        
        checkpoint_path = "/gscratch/dynamicsai/doris/timeVAE/final_results/model_SINDy_harmonic_osc_sigmoid_diff" + str(diff_term) + "_2stages_sparsity" + str(self.model.sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq_" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_diff_ics_mask_noL0norm_1000kl_xdotscale" #+ "_epoch" + str(epoch)
        print("checkpoint_path", checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    
        print("reconstruction", "I'm here")
        
        sindy_library = self.model.sindy_lib
        masked_sindy_coefficients = self.model.masked_sindy_coefficients
        result = self.model.result
        
        #self.model.encoder.save(checkpoint_path + "/encoder_simple_incomplete_epoch_" + str(epoch) + "_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_" + multipleTraj + "_mask" + ".keras")
        #self.model.decoder.save(checkpoint_path + "/decoder_simple_incomplete_epoch_" + str(epoch) + "_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_" + multipleTraj + "_mask" + ".keras")

        np.save(checkpoint_path + "/Epoch_" + str(epoch) + "_masked_sindy_harmonic_osc_sigmoid_2stages_sparsity" + str(self.model.sparsity) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_different_deriv_L2_" + multipleTraj + "_mask_xdotscale.npy", masked_sindy_coefficients)
        np.save(checkpoint_path + "/Epoch_" + str(epoch) + "_sindy_lib_harmonic_osc_sigmoid_2stages_sparsity" + str(self.model.sparsity) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_different_deriv_L2_" + multipleTraj + "_mask_xdotscale.npy", sindy_library)        
        np.save(checkpoint_path + "/Epoch_" + str(epoch) + "_X_dot_hat_harmonic_osc_sigmoid_2stages_sparsity" + str(self.model.sparsity) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_mask_xdotscale.npy", result)

#####################################################################################################
#####################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--std", type=float, default=1.0, required=False, help="std of SINDy coefficient")
args = parser.parse_args()

if __name__ == '__main__':

    sys.setrecursionlimit(10000)
    global len_seq_, latent_dim, batch_sz, sparsity, total_var_coeff, multipleTraj
    len_seq_= 1000
    diff_term = args.std #1.0
    batch_size = 1
    latent_dim = 2
    
    print("diff term", diff_term)
    sparsity = 10; #1(Fourier), 200;#50 - for different dataset (vs 200); #10 for smoothing at 8,20 #200 otherwise
    total_var_coeff = 1000;
    multipleTraj = "multipleTraj"
    
    #dataset = "harmonic_osc"
    #X = get_my_data(dataset)[0,:,:]
    #X = get_multiple_trajectories(X, len_seq=700) #parameter1
    
    #HARMONIC OSC
    #1 sinusoid
    #X = np.load("x_harmonic_osc_sinusoid_len_seq_" + str(len_seq_) + "_all_scaled_noMin.npy")
    #X_dot = np.load("x_dot_harmonic_sinusoid_len_seq_" + str(len_seq_) + "_all_scaled_noMin.npy")
    X_dot = np.load("x_dot_harmonic_osc_sigmoid_len_seq_1000_diff_max_xdot_new.npy")
    X = np.load("x_harmonic_osc_sigmoid_len_seq_1000_diff_max_xdot_new.npy")
    X_dot_norm = np.load("x_dot_harmonic_osc_sigmoid_len_seq_1000_normalized_new.npy")
    X_norm = np.load("x_harmonic_osc_sigmoid_len_seq_1000_normalized_new.npy")

    #X = np.load("x_harmonic_osc_2sinusoid_len_seq_" + str(len_seq_) + "_all_scaled_noMin.npy")
    #X_dot = np.load("x_dot_harmonic_2sinusoid_len_seq_" + str(len_seq_) + "_all_scaled_noMin.npy")

    #sigmoid
    #X = np.load("x_harmonic_osc_sigmoid_len_seq_2000_all_scaled_noMin.npy")
    #X_dot = np.load("x_dot_harmonic_sigmoid_len_seq_2000_all_scaled_noMin_v2.npy")
    #X = np.load("x_harmonic_osc_sigmoid_len_seq_2000_all_scaled_noMin_multipleTraj.npy")
    #X_dot = np.load("x_dot_harmonic_sigmoid_len_seq_2000_all_scaled_noMin_v2_multipleTraj.npy")
    #X_dot = np.load("x_dot_harmonic_sigmoid_len_seq_" + str(len_seq_) + "_all_scaled_noMin_diff" + str(diff_term) + "_multipleTraj_different_ic_diff_max_xdot.npy")
    #X = np.load("x_harmonic_osc_sigmoid_len_seq_" + str(len_seq_) + "_all_scaled_noMin_diff" + str(diff_term) + "_multipleTraj_different_ic_diff_max_xdot.npy")
    
    #X = np.load("x_harmonic_osc_sigmoid_len_seq_2000_all_scaled_noMin_diff1_multipleTraj_different_ic.npy")#[:1,:,:]
    #X_dot = np.load("x_dot_harmonic_sigmoid_len_seq_2000_all_scaled_noMin_diff1_multipleTraj_different_ic.npy")#[:1,:,:]
    
    #X = np.load("x_dot_harmonic_switch_signal_len_seq_2000_all_scaled_noMin_diff1_multipleTraj_different_ic_diff_max_xdot.npy")
    #X_dot = np.load("x_harmonic_osc_switch_signal_len_seq_2000_all_scaled_noMin_diff1_multipleTraj_different_ic_diff_max_xdot.npy")
    
    #X = np.load("x_dot_harmonic_switch_signal_len_seq_" + str(len_seq_)+ "_all_scaled_noMin_diff" + str(diff_term) + "_multipleTraj_different_ic_diff_max_xdot.npy")[:,:len_seq_,:]
    #X_dot = np.load("x_harmonic_osc_switch_signal_len_seq_" + str(len_seq_) + "_all_scaled_noMin_diff" + str(diff_term) + "_multipleTraj_different_ic_diff_max_xdot.npy")[:,:len_seq_,:]

    #X_dot = np.load("x_dot_lorenz_sigmoid_len_seq_2000_all_scaled_noMin_multipleTraj_different_ic.npy")
    #X = np.load("x_lorenz_sigmoid_len_seq_2000_all_scaled_noMin_multipleTraj_different_ic.npy")
    #X_dot = np.load("x_dot_lorenz_sigmoid_len_seq_2000_all_scaled_noMin_multipleTraj_different_ic_diff_max_Xonly.npy")
    #X = np.load("x_lorenz_sigmoid_len_seq_2000_all_scaled_noMin_multipleTraj_different_ic_diff_max_Xonly.npy")
    #X_dot = np.load("x_dot_lorenz_sigmoid_len_seq_10000_all_scaled_noMin_multipleTraj_different_ic_diff_max.npy")
    #X = np.load("x_lorenz_sigmoid_len_seq_10000_all_scaled_noMin_multipleTraj_different_ic_diff_max.npy")
    #X_dot = np.load("x_dot_lorenz_sigmoid_len_seq_4000_all_scaled_noMin_multipleTraj_different_ic_diff_max.npy")
    #X = np.load("x_lorenz_sigmoid_len_seq_4000_all_scaled_noMin_multipleTraj_different_ic_diff_max.npy")
    #X_dot = np.load("x_dot_lorenz_sigmoid_len_seq_6000_all_scaled_noMin_multipleTraj_different_ic_diff_max.npy")
    #X = np.load("x_lorenz_sigmoid_len_seq_6000_all_scaled_noMin_multipleTraj_different_ic_diff_max.npy")
    
    #X_dot = np.load("x_dot_lorenz_sharp_sigmoid_len_seq_1000_all_scaled_noMin_multipleTraj_different_icV2_diff_max_xdot.npy")
    #X = np.load("x_lorenz_sharp_sigmoid_len_seq_1000_all_scaled_noMin_multipleTraj_different_icV2_diff_max_xdot.npy")
    ##X_dot = np.load("x_dot_lorenz_sigmoid_len_seq_1000_all_scaled_noMin_multipleTraj_different_ic.npy")
    ##X = np.load("x_lorenz_sigmoid_len_seq_1000_all_scaled_noMin_multipleTraj_different_ic.npy")
    #X_dot_norm = np.load("x_dot_lorenz_sharp_sigmoid_len_seq_1000_all_scaled_noMin_multipleTraj_different_icV2_normalized.npy")
    #X_norm = np.load("x_lorenz_sharp_sigmoid_len_seq_1000_all_scaled_noMin_multipleTraj_different_icV2_normalized.npy")
    
    #random_switches
    #X_dot = np.load( "x_dot_harmonic_random_switches_len_seq_2500_all_scaled_noMin_diff1.0_multipleTraj_different_ic_diff_max_xdot.npy")
    #X = np.load( "x_harmonic_osc_random_switches_len_seq_2500_all_scaled_noMin_diff1.0_multipleTraj_different_ic_diff_max_xdot.npy")

    #Fourier series
    #X_dot = np.load("x_dot_harmonic_Fourier_series_freq1_len_seq_" + str(len_seq_) + "_all_scaled_noMin_diff" + str(diff_term) + "_multipleTraj_different_ic_diff_max_xdot.npy")       #[:,1200:,:]
    #X = np.load("x_harmonic_osc_Fourier_series_freq1_len_seq_" + str(len_seq_) + "_all_scaled_noMin_diff" + str(diff_term) + "_multipleTraj_different_ic_diff_max_xdot.npy")           #[:,1200:,:]

    #drift diffusion
    #X_dot = np.load("x_dot_harmonic_drift_diffusion_len_seq_3000_all_scaled_noMin_diff0.1_multipleTraj_different_ic.npy")[:,:2500,:]
    #X = np.load("x_harmonic_osc_drift_diffusion_len_seq_3000_all_scaled_noMin_diff0.1_multipleTraj_different_ic.npy")[:,:2500,:]

    len_seq_= 1000

    #lotka volterra
    #X_dot = np.load("data_x/x_dot_lorenz_simple_len_seq_1000_all_scaled_noMin_multipleTraj_different_icV2_diff_max_xdot.npy")[:,:,:2]
    #X = np.load("data_x/x_lorenz_simple_len_seq_1000_all_scaled_noMin_multipleTraj_different_icV2_diff_max_xdot.npy")[:,:,:2]
    #X_dot_norm = np.load("data_x/x_dot_lorenz_simple_len_seq_1000_all_scaled_noMin_multipleTraj_different_icV2_normalized.npy")[:,:,:2]
    #X_norm = np.load("data_x/x_lorenz_simple_len_seq_1000_all_scaled_noMin_multipleTraj_different_icV2_normalized.npy")[:,:,:2]
        
    print('data shape:', X.shape, "len_seq", len_seq_)
    N, T, D = X.shape
        
    idx = np.random.permutation(np.array(range(N)))
    X = X[idx, :, :]
    X_dot = X_dot[idx, :, :]
    X_norm = X[idx, :, :]
    X_dot_norm = X_dot[idx, :, :]
    
    lib_sz = library_size(D, poly_order=3)
    dt = 0.01
    
    #X_dot = compute_derivative(X, T, dt)
    vae = VariationalAutoencoderConv(
        seq_len=T,
        lib_sz = lib_sz,
        z_dim=D,
        latent_dim = latent_dim,
        batch_size = batch_size,
        hidden_layer_sizes=[5, 10],
    )
    
    """
    epoch_saved = 0
    vae.encoder = keras.models.load_model("final_results/model_SINDy_simple_incomplete_2stages_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq_" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_diff_ics_mask_noL0norm_1000kl_xmax_only/encoder_simple_incomplete_epoch_" + str(epoch_saved) + "_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_" + multipleTraj + "_mask" +".keras", custom_objects={'Sampling': Sampling})
    
    vae.decoder = keras.models.load_model("final_results/model_SINDy_simple_incomplete_2stages_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq_" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_diff_ics_mask_noL0norm_1000kl_xmax_only/decoder_simple_incomplete_epoch_" + str(epoch_saved) + "_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_" + multipleTraj + "_mask" + ".keras")
    """
    
    print("compile?")
    vae.compile(optimizer=Adam(decay=1e-5, clipnorm=1), run_eagerly=True)  
    vae.sparsity = sparsity; vae.total_var_coeff = total_var_coeff;
    # vae.summary() ; sys.exit()

    print("fit?")
    data = {}
    data["X"] = X; data["X_dot"] = X_dot; data["X_norm"] = X_norm; data["X_dot_norm"] = X_dot_norm; 
    r = vae.fit(data, epochs=30, batch_size=batch_size, shuffle=True, callbacks=[MyCallback()])
        
    total_loss_training = vae.total_loss_tracker.result()
    reconstruction_loss_training = vae.reconstruction_loss_tracker.result()
    np.save("/gsratch/dynamicsai/doris/timeVAE/final_results/total_loss_harmonic_osc_sigmoid_2stages_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size_" + str(batch_size) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_diff_ics_mask_noL0norm_1000kl_xdotscale.npy", total_loss_training)
    np.save("/gsratch/dynamicsai/doris/timeVAE/final_results/loss_recon_harmonic_osc_sigmoid_2stages_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size_" + str(batch_size) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_diff_ics_mask_noL0norm_1000kl_xdotscale.npy", reconstruction_loss_training)
    
    print("predict?")
    sindy_decoded_vae = vae.predict(data, batch_size=batch_size)
    sindy_decoded = vae.get_prior_samples(100)
    
    np.save("/gsratch/dynamicsai/doris/timeVAE/final_results/Last_sindy_prior_samples_harmonic_osc_sigmoid_2stages_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_diff_ics_mask_noL0norm_1000kl_xdotscale.npy", sindy_decoded[0])
    np.save("/gsratch/dynamicsai/doris/timeVAE/final_results/Last_sindy_vae_harmonic_osc_sigmoid_2stages_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_diff_ics_mask_noL0norm_1000kl_xdotscale.npy", sindy_decoded_vae)
    np.save("/gsratch/dynamicsai/doris/timeVAE/final_results/Last_x_dot_real_harmonic_osc_sigmoid_2stages_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_diff_ics_mask_noL0norm_1000kl_xdotscale.npy", X_dot)
    
    # compare original and posterior predictive (reconstructed) samples
    #draw_orig_and_post_pred_sample(X_dot, X_dot_decoded, n=5)

    z, _, _ = vae.encoder(X)
    sindy_coeffs = vae.decoder(z)
    sindy_coeffs = sindy_coeffs[0].numpy()
    #np.save("results/sindy_coeff_useless_harmonic_osc_2sinusoid_2stages_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size_" + str(batch_sz) + "_len_seq" + str(len_seq_) + ".npy", sindy_coeffs)
    
    # generate prior predictive samples by sampling from latent space
    #plot_latent_space(vae, 30, figsize=15)
    #plot_latent_space_timeseries(vae, 30, figsize=15)