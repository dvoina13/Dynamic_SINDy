#VAE with SINDy for Celegans, sparsity fixed (or not), total_var, can set var's to means and smoothen (with by convolving with padding)

import os, warnings, sys
from re import T
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np
import math

import tensorflow as tf
from tensorflow.keras.layers import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Permute, Input, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.backend import random_normal
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow.keras.regularizers as regularizers
import math

from utils2 import get_my_data, get_multiple_trajectories, draw_orig_and_post_pred_sample, plot_latent_space, library_size, sindy_library, normalize_SINDy, compute_derivative, plot_latent_space_timeseries, MinMaxScaler_Feat_Dim, MinMaxScaler
from vae_base_wSINDy_Celegans import BaseVariationalAutoencoder, Sampling
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
                    #activity_regularizer=regularizers.L2(0.01),
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
                    activity_regularizer=regularizers.L2(1),
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
                    activity_regularizer=regularizers.L2(1),
                    name=f'dec_deconv__{i+1}')(x)
        
        x4 = tf.identity(x)
        print("x5 decoder shape", x.shape)
        
        print("s flatten really a problem? decoder")
        x = Flatten(name='dec_flatten')(x)
        print("x6 decoder shape", x.shape)
        #x = Dense(self.seq_len * self.feat_dim, activation="relu", activity_regularizer=regularizers.L2(1), name="decoder_dense_third_final")(x)
        #x = Dense(self.seq_len * self.feat_dim, activation="relu", activity_regularizer=regularizers.L2(1), name="decoder_dense_second_final")(x)
        self.padding = 100
        x = Dense((self.seq_len+self.padding) * self.feat_dim, name="decoder_dense_final",)(x)
        print("x7 decoder shape", x.shape)
        self.decoder_outputs = Reshape(target_shape=(self.seq_len+self.padding, self.feat_dim))(x)

        print("shape self.decoder_outputs", self.decoder_outputs)
        self.decoder_outputs_mean = GlobalAveragePooling1D(data_format='channels_last')(self.decoder_outputs)
        
        #x = Permute((2,1))(Reshape(target_shape=((self.seq_len+self.padding), self.feat_dim))(x))
        #print("shape", x.shape)
        #self.decoder_outputs = Reshape(target_shape=(-1, self.seq_len+self.padding,1, 1))(x)
        #print("shape2", self.decoder_outputs)
        #self.decoder_outputs_smooth = tf.nn.conv1d(self.decoder_outputs, 1/int(padding//2)*tf.ones((int(padding)//2,1,1)), stride=[1,1,1], padding="SAME", data_format = "NWC")
        #print("self.decoder_outputs_smooth shape", self.decoder_outputs_smooth.shape)
        #self.decoder_outputs_smooth = Permute((2,1))(Reshape(target_shape=(self.feat_dim, self.seq_len+padding))(self.decoder_outputs_smooth))
        #self.decoder_outputs_smooth = self.decoder_outputs_smooth[:,int(padding//2):-int(padding//2), :]
        #print("decoder_outputs_smooth.shape", self.decoder_outputs_smooth.shape)
        
        x5 = tf.identity(self.decoder_outputs)
        print("x", self.decoder_outputs)
        print("x8 decoder shape", self.decoder_outputs.shape)
        
        decoder = Model(decoder_inputs, [self.decoder_outputs, self.decoder_outputs_mean, x1, x2, x3, x4, x5], name="decoder")
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
        if (self.model.epoch==0 and batch>=30 and batch%5==0) or (self.model.epoch>0 and batch%5==0):
            if len(self.model.masked_sindy_coefficients.shape) == 4:
                masked_sindy_coeff_reduced = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(self.model.masked_sindy_coefficients), axis=0), axis=0)
            
        
            self.model.mask = self.model.reconfigure_mask(masked_sindy_coeff_reduced, self.model.epoch)
        
        
        np.save("makes_sindy_at_batch_end.npy", self.model.masked_sindy_coefficients)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.model.epoch = epoch
        self.model.scaling = []
        
        
        
    def on_epoch_end(self, epoch, logs=None):
        
        print("I'm at the epoch end") 
        tf.print(tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(self.model.masked_sindy_coefficients), axis=0), axis=0), "hereee, self.masked_sindy_coefficients")
        
        if epoch>=100:
            self.sparsity = 1
            
        if len(self.model.masked_sindy_coefficients.shape) == 4:
            masked_sindy_coeff_reduced = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(self.model.masked_sindy_coefficients), axis=0), axis=0)
        print(masked_sindy_coeff_reduced, "masked_sindy_coeff_reduced")
        print("epoch before reconfigure", epoch)
        self.model.mask = self.model.reconfigure_mask(masked_sindy_coeff_reduced, epoch)
        
        checkpoint_path = "final_results/model_SINDy_Celegans_2stages_sparsity" + str(sparsity) + "_threshold_" + str(self.model.threshold) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq_" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_premask_noL0norm_1000kl_xdotscale_means4_smooth" #+ "_epoch" + 
        
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        #self.model.save(checkpoint_path, file_pref="h5")
        print("reconstruction", "I'm here")
        
        sindy_library = self.model.sindy_lib
        masked_sindy_coefficients = self.model.masked_sindy_coefficients
        result = self.model.result
        
        np.save(checkpoint_path + "/Epoch_" + str(epoch) + "_masked_sindy_Celegans_2stages_sparsity" + str(sparsity) + "_threshold_" + str(self.model.threshold) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_different_deriv_L2_" + multipleTraj + "_premask_xdotscale_means4_smooth.npy", masked_sindy_coefficients)
        np.save(checkpoint_path + "/Epoch_" + str(epoch) + "_sindy_lib_Celegans_2stages_sparsity" + str(sparsity) + "_threshold_"  + str(self.model.threshold) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_different_deriv_L2_" + multipleTraj + "_premask_xdotscale_means4_smooth.npy", sindy_library)        
        np.save(checkpoint_path + "/Epoch_" + str(epoch) + "_X_dot_hat_Celegans_2stages_sparsity" + str(sparsity)  + "_threshold_" + str(self.model.threshold) + "_total_var_" + str(total_var_coeff) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_premask_xdotscale_means4_smooth.npy", result)
#####################################################################################################
#####################################################################################################


if __name__ == '__main__':

    sys.setrecursionlimit(10000)
    global len_seq_, latent_dim, batch_sz, sparsity, total_var_coeff, multipleTraj
    len_seq_= 2000
    batch_size = 1
    latent_dim = 2
    
    sparsity = 0 #0 #- for different dataset (vs 200); #10 for smoothing at 8,20 #200 otherwise
    total_var_coeff = 1000;
    multipleTraj = ""
    
    #X = np.load("2pca_x_smooth_neural_act_worm4_x_dot_3d_noMin.npy")[:,:len_seq_,:]
    #X_dot = np.load("2pca_x_dot_smooth_neural_act_worm4_x_dot_3d_noMin.npy")[:,:len_seq_,:]
    X = np.load("x_Celegans_worm4_all_scaled_noMin.npy")
    X_dot = np.load("x_dot_Celegans_worm4_all_scaled_noMin.npy")
    #X = np.load("2pca_x_smooth_neural_act_worm4_x_dot_3d_nonNormalized.npy")[:,:len_seq_,:]
    #X_dot = np.load("2pca_x_dot_smooth_neural_act_worm4_x_dot_3d_nonNormalized.npy")[:,:len_seq_,:]
    X[:,:,[1,2]] = X[:,:,[2,1]]; X_dot[:,:,[1,2]] = X_dot[:,:,[2,1]];
    X = X[:,:len_seq_,:2]; X_dot = X_dot[:,:len_seq_,:2]
    print('data shape:', X.shape)
    N, T, D = X.shape
        
    idx = np.random.permutation(np.array(range(N)))
    X = X[idx, :, :]
    X_dot = X_dot[idx, :, :]
    
    lib_sz = library_size(D, poly_order=3, include_constant=True)
    dt = 0.35749752
    
    #X_dot = compute_derivative(X, T, dt)
    vae = VariationalAutoencoderConv(
        seq_len=T,
        lib_sz = lib_sz,
        z_dim=D,
        latent_dim = latent_dim,
        batch_size = batch_size,
        hidden_layer_sizes=[5, 10],
    )
    
    #print("add weight decay")
    #vae = add_weight_decay(vae)
    print("compile?")
    vae.compile(optimizer=Adam(decay=1e-5, clipnorm=1), run_eagerly=True) #
    vae.sparsity = sparsity; vae.total_var_coeff = total_var_coeff;
    # vae.summary() ; sys.exit()

    print("fit?")
    data = {}
    data["X"] = X; data["X_dot"] = X_dot;
    r = vae.fit(data, epochs=500, batch_size=batch_size, shuffle=True, callbacks=[MyCallback()])
        
    total_loss_training = vae.total_loss_tracker.result()
    reconstruction_loss_training = vae.reconstruction_loss_tracker.result()
    
    np.save("final_results/total_loss_Celegans_2stages_sparsity" + str(sparsity)  + "_threshold_" + str(vae.threshold) + "_total_var_" + str(total_var_coeff) + "_batch_size_" + str(batch_size) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_premask_noL0norm_1000kl_xdotscale_means4_smooth.npy", total_loss_training)
    np.save("final_results/loss_recon_Celegans_2stages_sparsity" + str(sparsity) + "_threshold_" + str(vae.threshold) + "_total_var_" + str(total_var_coeff) + "_batch_size_" + str(batch_size) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_premask_noL0norm_1000kl_xdotscale_means4_smooth.npy", reconstruction_loss_training)
    
    print("predict?")
    sindy_decoded_vae = vae.predict(data, batch_size=batch_size)
    sindy_decoded = vae.get_prior_samples(100)
    
    np.save("final_results/Last_sindy_prior_samples_Celegans_2stages_sparsity" + str(sparsity) + "_threshold_" + str(vae.threshold) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_premask_noL0norm_1000kl_xdotscale_means4_smooth.npy", sindy_decoded[0])
    np.save("final_results/Last_sindy_vae_Celegans_2stages_sparsity" + str(sparsity) + "_threshold_" + str(vae.threshold) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_premask_noL0norm_1000kl_xdotscale_means4_smooth.npy", sindy_decoded_vae)
    np.save("final_results/Last_x_dot_real_Celegans_2stages_sparsity"  + "_threshold_" + str(vae.threshold) + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size" + str(batch_size) + "_len_seq" + str(len_seq_) + "_latent_dim_" + str(latent_dim) + "_different_deriv2_L2_" + multipleTraj + "_premask_noL0norm_1000kl_xdotscale_means4_smooth.npy", X_dot)
    
    
    # compare original and posterior predictive (reconstructed) samples
    #draw_orig_and_post_pred_sample(X_dot, X_dot_decoded, n=5)

    z, _, _ = vae.encoder(X)
    sindy_coeffs = vae.decoder(z)
    sindy_coeffs = sindy_coeffs[0].numpy()
    #np.save("results/sindy_coeff_useless_harmonic_osc_2sinusoid_2stages_sparsity" + str(sparsity) + "_total_var_" + str(total_var_coeff) + "_batch_size_" + str(batch_sz) + "_len_seq" + str(len_seq_) + ".npy", sindy_coeffs)
    
    # generate prior predictive samples by sampling from latent space
    #plot_latent_space(vae, 30, figsize=15)
    #plot_latent_space_timeseries(vae, 30, figsize=15)