
import os, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

from abc import ABC, abstractmethod
import numpy as np
import math
import tensorflow as tf
import joblib 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal
from tensorflow.keras.layers import Reshape, Permute
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Permute, Input, Concatenate, GlobalAveragePooling1D

import derivative
from scipy.signal import savgol_filter

from utils2 import sindy_library, normalize_SINDy

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def integrate(x, X_dot_hat, N, dt):
    
    trajectory = [x]
    print("N", N)
    
    for j in range(N-1):
        x = x + dt*X_dot_hat[0,j,:]
        trajectory.append(x)
        
    return np.array(trajectory)

def normalize_sindy(sindy_coeff):
     folder = "/home/doris/timeVAE"
     x_max = np.load(folder + "/x_smooth_worm4_3d_maxes.npy")
     x_dot_max = np.load(folder + "/x_dot_smooth_worm4_3d_maxes.npy")
     
     print("in normalized_sindy", sindy_coeff.shape)
     normalized_sindy = {}
     normalized_sindy["d"] = sindy_coeff[-1,:,0,1]*x_dot_max[2]
     normalized_sindy["c"] = sindy_coeff[-1,:,1,1]*x_dot_max[2]/x_max[0]
     normalized_sindy["gamma"] = sindy_coeff[-1,:,2,1]*x_dot_max[2]/(x_dot_max[0])
     normalized_sindy["b"] = sindy_coeff[-1,:,3,1]*x_dot_max[2]/(x_max[0]**2)
     normalized_sindy["a"] = sindy_coeff[-1,:,6,1]*x_dot_max[2]/(x_max[0]**3)
   
     return normalized_sindy
    
def compute_z_loss(sindy_coefficients, x):
    
    normalized_sindy = normalize_sindy(sindy_coefficients)
    
    a = normalized_sindy["a"]
    b = normalized_sindy["b"]
    c = normalized_sindy["c"]
    gamma = normalized_sindy["gamma"]
    N = a.shape
    
    print("a,b,c,gamma", a,b,c,gamma)
    dt = 0.35749752
    #int_val = np.squeeze(np.array(gamma**2 - 4*(3*a*x**2 + 2*b*x + c)))
    #int_val = [np.sqrt(int_val[i] + 0j) for i in range(2000)]
    int_val = tf.compat.v1.math.sqrt(tf.cast(gamma**2 - 4*(3*a*x**2 + 2*b*x + c), dtype=tf.complex128))
    print("int_val", int_val)
    lambda1 = 1/2*(tf.cast(gamma, dtype=tf.complex128) -  int_val)
    lambda2 = 1/2*(tf.cast(gamma, dtype=tf.complex128) +  int_val)

    z1 = 1 + dt*lambda1
    z2 = 1 + dt*lambda2
    
    print("z1, z2", z1, z2)
    z1_norm = tf.math.real(z1)**2 + tf.math.imag(z1)**2
    z2_norm = tf.math.real(z2)**2 + tf.math.imag(z2)**2
                       
    print("z1_norm, z2_norm", z1_norm, z2_norm)
    return [z1_norm, z2_norm]

def hard_sigmoid(x):
    return tf.math.minimum( tf.math.maximum(x, tf.zeros_like(x)), tf.ones_like(x))

class _L0_Norm(tf.keras.layers.Layer):
    def __init__(self, lib_sz, z_dim, T=None, loc_mean=0, loc_sdev=0.01,
                 beta=2 / 3, gamma=-0.1,
                 zeta=1.1, fix_temp=True, **kwargs):
        
        super(_L0_Norm, self).__init__(**kwargs)        
        
        self.lib_sz = lib_sz
        self.z_dim = z_dim
        self.T = T
        
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)
        
        self.loc_mean = loc_mean
        self.loc_sdev = loc_sdev
        self.beta = beta
        self.training = True
        self.fix_temp = fix_temp

    def build(self, shape):
        
        library_dim = self.lib_sz;
        z_dim = self.z_dim;
        T = self.T
        self.loc = self.add_weight(shape=(library_dim, z_dim), #self.T
                        initializer=tf.keras.initializers.RandomNormal(mean=self.loc_mean, stddev=self.loc_sdev, seed=None),                             trainable=True)

        if self.fix_temp==False:
            self.temp = self.add_weight(shape=(1), initializer=tf.keras.initializers.Constant(value=self.beta), trainable = True)
        else:
            self.temp = self.beta
            
        print("self.loc, self.temp", self.loc.shape, self.temp)
        super(_L0_Norm, self).build(shape)

    def call(self, input_l0):
        if self.training:
            
            u_d = tfp.distributions.Uniform(low=0.0, high=1.0)
            u = tfp.distributions.Sample(u_d, sample_shape=(self.T, self.lib_sz, self.z_dim)).sample()#self.T
            
            s = tf.math.sigmoid((tf.math.log(u) - tf.math.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty =  tf.math.reduce_sum(tf.math.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio))#.sum(1).mean()
        else:
            s = tf.math.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
            
        print("s call at end", s.shape)
        
        return hard_sigmoid(s), penalty
        

class BaseVariationalAutoencoder(Model, ABC):
    def __init__(self,  
            seq_len, 
            lib_sz,
            z_dim,
            latent_dim,
            batch_size,
            dt =  0.35749752,
            threshold = 0.01,
            reconstruction_wt = 3.0,
            alpha = 1e-5,
            sparsity = None,
            total_var_coeff = None,
            **kwargs  ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.lib_sz = lib_sz
        self.z_dim = z_dim
        self.feat_dim = lib_sz * z_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.dt = dt
        self.reconstruction_wt = reconstruction_wt
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean( name="reconstruction_loss" )
        self.kl_loss_tracker = Mean(name="kl_loss")
        
        self.epoch = 0
        self.batch = 0
        
        self.threshold = threshold
        self.mask = tf.Variable(tf.ones([self.batch_size, self.seq_len, self.lib_sz,self.z_dim], tf.float32), trainable=False)
        self.l0 = _L0_Norm(self.lib_sz, self.z_dim)  #####
        self.alpha = alpha

        self.encoder = None
        self.decoder = None
        
        self.sparsity = sparsity
        self.total_var_coeff = total_var_coeff
        
        self.use_l0 = False
        self.switch_loss = False
        self.smooth_sindy = False
        
    def call(self, data):
        X = data["X"]

        self.l0.training = False
        
        z_mean, _, z = self.encoder(X)
        decoder_output, x1, x2, x3, x4, x5 = self.decoder(z)
        
        X = Reshape(target_shape=(self.seq_len, self.z_dim))(X)
        decoder_output = Reshape(target_shape=(self.seq_len, self.feat_dim))(decoder_output)
        self.sindy_coefficients = Reshape(target_shape=(-1, self.lib_sz, self.z_dim))(decoder_output)
        
        self.masked_sindy_coefficients = tf.math.multiply(self.sindy_coefficients, self.mask)
        
        if self.use_l0:
           l0_mask, penalty = self.l0(self.lib_sz)
           #print("l0 maskk shape", l0_mask.shape, l0.T)
           #self.masked_sindy_coefficients *= tf.repeat(Reshape(target_shape=(1, self.lib_sz, self.z_dim))(l0_mask), repeats=[l0.T,0,0], axis=0)
           self.masked_sindy_coefficients *= l0_mask
            
        sindy_lib = Reshape(target_shape=(self.seq_len, 1, self.lib_sz))(sindy_library(X, poly_order=3, include_constant=True))
        X_dot_hat = Reshape(target_shape=(self.seq_len, self.z_dim))(tf.linalg.matmul(sindy_lib, self.masked_sindy_coefficients))
            
        if len(X_dot_hat.shape) == 1: X_dot_hat = X_dot_hat.reshape((1, -1))
        
        self.result = X_dot_hat
        
        return X_dot_hat
            
    def reconfigure_mask(self, sindy_coeffs, epoch):
        """
        mask1, mask2 = tf.unstack(self.mask, axis = 3)
        print("mask1, mask2 shape", mask1.shape, mask2.shape)
        mask1 = np.array(tf.unstack(mask1)); mask2 = np.array(tf.unstack(mask2))
        print("mask1, mask2 shape", mask1.shape, mask2.shape)
        mask_numpy = np.concatenate((np.expand_dims(mask1, axis=3), np.expand_dims(mask2, axis=3)), axis=3)
        print("mask numpy shape", mask_numpy.shape)
        #mask_numpy = 0*mask_numpy
        mask_numpy[:,:,:,0] = 0
        mask_numpy[:,:,2,0] = 1
        mask_numpy[:,:,:,1] = 0
        mask_numpy[:,:,0,1] = 1
        mask_numpy[:,:,1,1] = 1
        mask_numpy[:,:,2,1] = 1
        mask_numpy[:,:,3,1] = 1
        mask_numpy[:,:,6,1] = 1
        
        mask1 = mask_numpy[:,:,:,0]; mask2 = mask_numpy[:,:,:,1];
        self.mask = tf.stack([mask1, mask2], axis=3)
        print("self.mask shape", self.mask.shape)
        print("Self.mask", self.mask[0,0,:,:]) 
        """

        sindy_coeffs_abs_mean = sindy_coeffs
        inds = tf.where(sindy_coeffs_abs_mean < self.threshold)

        tf.print("sindy_coeffs.shapeeee", sindy_coeffs.shape)
        tf.print(sindy_coeffs_abs_mean.shape, "sindy_coeffs_abs_mean")   
        tf.print(sindy_coeffs_abs_mean, "sindy_coeffs_abs_mean")
        print("threshold", self.threshold)
        print("inds", inds)
        print("Ill write this up soon")
        tf.print(self.mask.shape, "self.mask shape")
        print("epoch after reconfigure", epoch)
        
        if epoch>800:
            self.switch_loss = True
        print("self.switch_loss", self.switch_loss)
        
        if epoch < 100:
            mask_lol = tf.unstack(self.mask, axis = 3)
            print("hello1", len(mask_lol))
            print("hello2", mask_lol[0].shape)
            mask1 = mask_lol[0]; mask2 = mask_lol[1]
            print("mask1,2 initial shape", mask1.shape, mask2.shape)
            mask1 = np.array(tf.unstack(mask1)); mask2 = np.array(tf.unstack(mask2))
            mask_numpy = np.concatenate((np.expand_dims(mask1, axis=3), np.expand_dims(mask2, axis=3)), axis=3)
            print(mask_numpy.shape, np.array(inds).shape)
            print("whatt", np.array(inds))
            mask_numpy[:,:,:,0] = 0; mask_numpy[:,:,2,0] = 1; 
            print("types", type(mask1), type(mask2))
            mask1 = mask_numpy[:,:,:,0]; mask2 = mask_numpy[:,:,:,1];
        
            self.mask = tf.stack([mask1, mask2], axis=3)
            
        if epoch>=100 and epoch%10==0:
            mask_lol = tf.unstack(self.mask, axis = 3)
            print("hello1", len(mask_lol))
            print("hello2", mask_lol[0].shape)
            mask1 = mask_lol[0]; mask2 = mask_lol[1]
            print("mask1,2 initial shape", mask1.shape, mask2.shape)
            mask1 = np.array(tf.unstack(mask1)); mask2 = np.array(tf.unstack(mask2))
            mask_numpy = np.concatenate((np.expand_dims(mask1, axis=3), np.expand_dims(mask2, axis=3)), axis=3)
            print(mask_numpy.shape, np.array(inds).shape)
            print("whatt", np.array(inds))
            mask_numpy[:,:,np.array(inds[:,0]), np.array(inds[:,1])] = 0
            mask_numpy[:,:,:,0] = 0; mask_numpy[:,:,2,0] = 1; 
            print("types", type(mask1), type(mask2))
            mask1 = mask_numpy[:,:,:,0]; mask2 = mask_numpy[:,:,:,1];
        
            self.mask = tf.stack([mask1, mask2], axis=3)
            
        #if epoch>100 and epoch%10==0:
        #    self.sparsity += 2
       
        print("final mask", self.mask[0,0,:,:])        
        
        return self.mask   
    
    def get_num_trainable_variables(self):
        trainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.trainable_weights]))
        nonTrainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights]))
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams


    def get_prior_samples(self, num_samples):
        Z = np.random.randn(num_samples, self.latent_dim)
        samples = self.decoder.predict(Z)
        return samples
    

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        if type(samples) is list:
            return samples[0]
        else:
            return samples

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    
    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    #@abstractmethod
    #def get_derivative(self, **kwargs):
    #    raise NotImplementedError
        
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        
    def _get_reconstruction_loss(self, X_dot, X_dot_hat): 

        def get_reconst_loss_by_axis(X, X_c, axis): 
            x_r = tf.reduce_mean(X, axis = axis)
            x_c_r = tf.reduce_mean(X_c, axis = axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall    
        tf.print(tf.where(tf.math.is_nan(X_dot_hat)), "is nan in X_dot_hat!")

        err = tf.math.squared_difference(X_dot, X_dot_hat)
        #tf.print(err, "err")
        #tf.print(err.shape, "err shape")
        #lol = tf.math.is_nan(err)
        #tf.print(tf.math.is_nan(err), "is nan")
        #tf.print("lol shape", lol.shape)
        #tf.print(tf.where(tf.math.is_nan(err)), "important!")
        reconst_loss = tf.reduce_sum(err)
        #tf.print(reconst_loss, "reconst_loss")  
    
        reconst_loss += get_reconst_loss_by_axis(X_dot, X_dot_hat, axis=[2])     # by time axis        
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis
        print("final line i reconst loss")
        return reconst_loss
    
    def train_step(self, data):
        X = data[0]["X"]
        X_dot = data[0]["X_dot"]
        
        print("I'm training!")
        print("sparsityyyyyyyyyyy", self.sparsity)
        print("total_var", self.total_var_coeff)
        
        N = X.shape[1]
        dt = 0.35749752

        self.l0.training = True
        self.l0.T = X.shape[1]; T = X.shape[1];
        weight_decay_l0 = 1e-5;
        print("T", self.l0.T)
        print("self.l0", self.l0)
        
        loss_arr = []; recon_loss_arr = []; kl_loss_arr = [];
        with tf.GradientTape() as tape:
            
            z_mean, z_log_var, z = self.encoder(X)
            decoder_output, decoder_output_mean, x1, x2, x3, x4, x5 = self.decoder(z)
            decoder_hidden_act = [x1, x2, x3, x4, x5]
            
            print("z_mean, z_log_var, decoder_output shapes", z_mean.shape, z_log_var.shape, decoder_output.shape)
            #X = Reshape(target_shape=(self.seq_len, self.z_dim+10))(X)
            X = Reshape(target_shape=(self.seq_len, self.z_dim))(X)
            print("X shape", X.shape) 
            decoder_output = Reshape(target_shape=(self.seq_len + self.padding, self.feat_dim))(decoder_output)

            mask = np.zeros((self.batch_size, self.seq_len, self.lib_sz, self.z_dim))
            mask[:,:,1,1] = 1
            mask[:,:,2,1] = 1
            mask[:,:,3,1] = 1
            mask[:,:,6,1] = 1
            mask[:,:,2,0] = 1
            
            #smoothness stuff
            
            if self.epoch > 0:
                decoder_output = Permute((2,1))(decoder_output)
                print("shape", decoder_output.shape)
                decoder_output = Reshape(target_shape=(-1, 1, self.seq_len+self.padding, 1))(decoder_output)
                print("shape2", decoder_output.shape)
                decoder_output_smooth = tf.nn.conv1d(decoder_output, 1/int(self.padding//2)*tf.ones((int(self.padding)//2,1,1)), stride=[1,1,1], padding="SAME", data_format = "NWC")
                print("self.decoder_outputs_smooth shape", decoder_output_smooth.shape)
                decoder_output_smooth = Permute((2,1))(Reshape(target_shape=(self.feat_dim, self.seq_len+self.padding))(decoder_output_smooth))
                decoder_output = decoder_output_smooth[:,int(self.padding//2):-int(self.padding//2), :]
                print("padding problem?", decoder_output.shape)
            else:
                decoder_output = decoder_output[:,int(self.padding//2):-int(self.padding//2), :]
                print("padding problem?", decoder_output.shape)


            decoder_output_mean = Reshape(target_shape=(self.lib_sz, self.z_dim))(decoder_output_mean)
            decoder_output = Reshape(target_shape=(self.seq_len, self.lib_sz, self.z_dim))(decoder_output)
            decoder_output_mean = tf.repeat(decoder_output_mean, self.seq_len, axis=0)            
            decoder_output_mean = tf.expand_dims(decoder_output_mean, axis = 0)
            print("decoder outputs mean shape", decoder_output_mean.shape)
        #print("decoder_outputs_smooth.shape", self.decoder_outputs_smooth.shape)
            
            """
            print("batch!", self.batch, self.smooth_sindy)
            if self.epoch>=0 and self.smooth_sindy:
                        #differentiator = derivative.Kalman(alpha=0.5)
                        decoder_smooth = Reshape(target_shape=(-1, self.seq_len))(Permute((2,1))(decoder_output))
                        print("decoder_diff.shapeee", decoder_smooth.shape)
                        decoder_smooth = savgol_filter(decoder_smooth, 201, 3)#501
                        print(savgol_filter(decoder_smooth, 201, 3).shape)
                        print("lolz", self.batch_size, self.feat_dim, self.seq_len)
                        print("decoder_diff.shapeee", decoder_smooth.shape)

                        #decoder_smooth = Permute((2,1))( Reshape( target_shape=(-1, self.feat_dim, self.seq_len))(differentiator.x(np.squeeze(decoder_smooth), np.arange(0,self.dt*self.seq_len, self.dt))))
                        print("decoder_diff.shapeee", decoder_smooth.shape)

                        decoder_smooth = Permute((2,1))(decoder_smooth) # window size 51, polynomial order 3
                        decoder_output = decoder_smooth
                        print("decoder_diff.shapeee", decoder_smooth.shape)
            """
            decoder_output_ = mask*decoder_output_mean + (1-mask)*decoder_output;

            self.sindy_coefficients = Reshape(target_shape=(-1, self.lib_sz, self.z_dim))(decoder_output_)
            print("decoder_output, sindy_coefficients shape", decoder_output.shape, self.sindy_coefficients.shape)
            print("mask shape", self.mask.shape)
            self.masked_sindy_coefficients = tf.math.multiply(self.sindy_coefficients, self.mask)
            
            #l0_norm_sindy = tf.math.reduce_mean(tf.math.abs(tf.reshape(self.masked_sindy_coefficients, [-1])))
            #print("partial norm", tf.squeeze(self.masked_sindy_coefficients).shape, tf.norm(tf.squeeze(self.masked_sindy_coefficients), ord='euclidean', axis=0), tf.norm(tf.squeeze(self.masked_sindy_coefficients), ord='euclidean', axis=0).shape)
            
            l0_norm_sindy = tf.math.reduce_mean(tf.norm(tf.squeeze(self.masked_sindy_coefficients + np.finfo(np.float32).eps), ord=1, axis=0))
            total_var = tf.math.reduce_mean(tf.math.abs(self.masked_sindy_coefficients[:,1:,:,:] - self.masked_sindy_coefficients[:,:-1,:,:]))/tf.math.reduce_mean(tf.math.abs(self.masked_sindy_coefficients) + np.finfo(np.float32).eps)
            #total_var = tf.math.reduce_mean(tf.math.abs(self.masked_sindy_coefficients[:,:-4,:,:] - 8*self.masked_sindy_coefficients[:,1:-3,:,:] + 8*self.masked_sindy_coefficients[:,3:-1,:,:] - self.masked_sindy_coefficients[:,4:,:,:]))
            total_var_x2 = tf.math.reduce_mean(tf.math.abs(x2[:,1:,:] - x2[:,:-1,:]))
            total_var_x3 = tf.math.reduce_mean(tf.math.abs(x3[:,1:,:] - x3[:,:-1,:]))
            total_var_x4 = tf.math.reduce_mean(tf.math.abs(x4[:,1:,:] - x4[:,:-1,:]))
            print("l0_norm_sindy", l0_norm_sindy)
            
            if self.use_l0:
                l0_mask, penalty = self.l0(self.lib_sz)
                print("~~~l0 loc~~~", self.l0.loc)
                print("l0_mask_shape, self.masked_sindy_coeffs.shape", l0_mask.shape, self.masked_sindy_coefficients.shape)
                #print("!Q", tf.expand_dims(tf.tile(tf.expand_dims(l0_mask, axis=0), tf.constant([self.l0.T,1,1], tf.int32)), axis=0).shape)
                #self.masked_sindy_coefficients *= tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(l0_mask, axis=0), tf.constant([self.l0.T,1,1], tf.int32)), axis=0), tf.constant([self.batch_size,1,1,1], tf.int32))                
                self.masked_sindy_coefficients *= l0_mask
                
            print("self.maskedd_sindy_coeffs", tf.math.reduce_mean(tf.math.reduce_mean(tf.math.abs(self.masked_sindy_coefficients), axis=0), axis=0))
            print("X shape", X.shape) 

            normalized_lib, scaling = normalize_SINDy(sindy_library(X[:,:,:3], poly_order=3, include_constant=True))
            sindy_lib = Reshape(target_shape=(self.seq_len, 1, self.lib_sz))(normalized_lib)#(sindy_library(X[:,:,:2], poly_order=3))#(normalized_lib)
            self.sindy_lib = sindy_lib; self.scaling.append(scaling);
            #print("scaling", scaling)
            X_dot_hat = Reshape(target_shape=(self.seq_len, self.z_dim))(tf.linalg.matmul(sindy_lib, self.masked_sindy_coefficients))
            
            #X_hat = integrate(X[0,0,:], X_dot_hat, N, dt)
            self.result = X_dot_hat  #X_dot_hat
            
            reconstruction_loss = self._get_reconstruction_loss(X_dot, X_dot_hat)
            
            #if np.isnan(X_hat.mean()):
            #    reconstruction_loss2 = np.inf
            #else: 
            #    reconstruction_loss2 = self._get_reconstruction_loss(X, np.expand_dims(X_hat,0)) #(X_dot, X_dot_hat)
                
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))

            """
            z_loss = compute_z_loss(self.masked_sindy_coefficients, X[:,:,0])
            z1_loss = tf.math.reduce_max(z_loss[0]); z2_loss = tf.math.reduce_max(z_loss[1])
            print("z_loss", z1_loss, z2_loss)
 
            z_loss = tf.cast(tf.math.sigmoid(1000*(z1_loss-1)), dtype=tf.float32) + tf.cast(tf.math.sigmoid(1000*(z2_loss-1)), dtype=tf.float32)
            print("z_loss", z_loss)
            """
            
            if self.use_l0:
                total_loss = self.reconstruction_wt * reconstruction_loss + 1000*kl_loss + self.total_var_coeff*total_var + weight_decay_l0 * penalty + self.sparsity*l0_norm_sindy #+ total_var_x2 + total_var_x3 + total_var_x4#
            else:
                penalty = 0
                total_loss = self.reconstruction_wt * reconstruction_loss + 1000*kl_loss + self.total_var_coeff*total_var + self.sparsity*l0_norm_sindy #+ 100*z_loss#+ 1000*total_var_x2 + 1000*total_var_x3 + 1000*total_var_x4# 
                
            if self.switch_loss:
                print("switch_loss", self.switch_loss)
                total_loss = self.reconstruction_wt * reconstruction_loss + 1000*kl_loss + self.total_var_coeff*total_var #+ 1000*total_var_x2 + 1000*total_var_x3 + 100*total_var_x4
                
            print("total_var, recon_loss, kl_loss, l0_norm_sindy, penalty", total_var, reconstruction_loss, kl_loss, l0_norm_sindy, penalty)
            print("total_var, recon_loss, kl_loss", self.total_var_coeff*total_var, self.reconstruction_wt * reconstruction_loss, 1000*kl_loss)

            loss_arr.append(total_loss);
            recon_loss_arr.append(reconstruction_loss);
            kl_loss_arr.append(kl_loss_arr)
            
            print("total loss", total_loss, "200")
        
        #print("masked sindy before", self.masked_sindy_coefficients)
        #dy_dx = tape.gradient(total_loss, z_loss)
        #print("GRADIENT", dy_dx.numpy())

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #print("grads", grads)
        #print("masked sindy after", self.masked_sindy_coefficients)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        for i, x in enumerate(decoder_hidden_act):
            np.save("decoder_output_x" + str(i), x)
            
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            #"loss_arr": loss_arr,
            #"recon_loss_arr": recon_loss_arr,
            #"kl_loss_arr": kl_loss_arr
        }
    
    def test_step(self, data): 
        print("do I even use this")
        X = data[0]["X"]
        X_dot = data[0]["X_dot"]
        
        self.l0.training = False
        
        z_mean, z_log_var, z = self.encoder(X)
        decoder_output, x1, x2, x3, x4, x5 = self.decoder(decoder_input)
        
        X = Reshape(target_shape=(self.seq_len, self.z_dim))(X)
        decoder_output = Reshape(target_shape=(self.seq_len, self.feat_dim))(decoder_output)
        self.sindy_coefficients = Reshape(target_shape=(-1, self.lib_sz, self.z_dim))(decoder_output)
        
        #self.mask = self.reconfigure_mask(self.sindy_coefficients)
        self.masked_sindy_coefficients = tf.math.multiply(self.sindy_coefficients, self.mask)

        if self.use_l0:
           l0_mask, penalty = self.l0(self.lib_sz)
           
           #self.masked_sindy_coefficients *= l0_mask
           self.masked_sindy_coefficients *= tf.repeat(Reshape(target_shape=(1, self.lib_sz, self.z_dim))(l0_mask), repeats=[l0.T,0,0], axis=0)
         
        
        sindy_lib = Reshape(target_shape=(self.seq_len, 1, self.lib_sz))(sindy_library(X, poly_order=3, include_constant=True))
        X_dot_hat = Reshape(target_shape=(self.seq_len, self.z_dim))(tf.linalg.matmul(sindy_lib, self.masked_sindy_coefficients))
            
        self.result = X_dot_hat
        
        #tf.compat.v1.enable_eager_execution()
        sindy_coef = self.sindy_coefficients.numpy()
        tf.print("sindy_coef", sindy_coef)
        
        reconstruction_loss = self._get_reconstruction_loss(X_dot, X_dot_hat)
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
        # kl_loss = kl_loss / self.latent_dim

        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    def save_weights(self, model_dir, file_pref): 
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(encoder_wts, os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        joblib.dump(decoder_wts, os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

    
    def load_weights(self, model_dir, file_pref):
        encoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        decoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)


    def save(self, model_dir, file_pref): 

        self.save_weights(model_dir, file_pref)
        dict_params = {

            'seq_len': self.seq_len,
            'feat_dim': self.feat_dim,
            'latent_dim': self.latent_dim,
            'reconstruction_wt': self.reconstruction_wt,
            'hidden_layer_sizes': self.hidden_layer_sizes,
        }
        params_file = os.path.join(model_dir, f'{file_pref}parameters.pkl') 
        joblib.dump(dict_params, params_file)


#####################################################################################################
#####################################################################################################


if __name__ == '__main__':

    pass