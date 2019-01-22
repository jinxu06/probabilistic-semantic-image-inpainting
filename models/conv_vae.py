import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d, down_shift, right_shift
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shifted_deconv2d, up_left_shifted_deconv2d, up_shift, left_shift
from blocks.samplers import gaussian_sampler, mix_logistic_sampler, bernoulli_sampler
from blocks.estimators import compute_gaussian_kld, estimate_mmd
from blocks.losses import mix_logistic_loss, bernoulli_loss
from blocks.helpers import int_shape, get_name, broadcast_masks_tf
from blocks.metrics import bits_per_dim_tf

class ConvVAE(object):

    def __init__(self, counters={}):
        self.counters = counters

    def build_graph(self, batch_size, img_size, z_dim, reg_type, beta=1.0, network_type="large", sample_range=1.0, nonlinearity=tf.nn.elu, bn=True, kernel_initializer=None, kernel_regularizer=None, nr_logistic_mix=10):

        self.batch_size = batch_size
        self.img_size = img_size
        self.reg_type = reg_type
        self.beta = beta
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bn = bn
        self.sample_range = sample_range
        self.nr_logistic_mix = nr_logistic_mix
        self.network_type = network_type
        self.__model()
        self.__loss()

    def __model(self):
        print("******   Building Graph   ******")
        # placeholders
        if self.network_type == 'binary':
            self.num_channels = 1
        else:
            self.num_channels = 3
        self.x = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size, self.num_channels))
        self.x_bar = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size, self.num_channels))
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.dropout_p = tf.placeholder(tf.float32, shape=())
        self.masks = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size))
        self.input_masks = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size))
        # choose network size
        if self.img_size == 32:
            if self.network_type == 'large':
                encoder = conv_encoder_32_large_bn
                decoder = conv_decoder_32_large_mixture_logistic
            else:
                raise Exception("unknown network type")
        elif self.img_size == 28:
            if self.network_type == 'binary':
                encoder = conv_encoder_28_binary
                decoder = conv_decoder_28_binary
            else:
                raise Exception("unknown network type")
        kwargs = {
            "nonlinearity": self.nonlinearity,
            "bn": self.bn,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "is_training": self.is_training,
            "counters": self.counters,
        }
        with arg_scope([encoder, decoder], **kwargs):
            inputs = self.x
            inputs = inputs * broadcast_masks_tf(self.masks, num_channels=self.num_channels)
            inputs = tf.concat([inputs, broadcast_masks_tf(self.masks, num_channels=1)], axis=-1)
            self.z_mu, self.z_log_sigma_sq = encoder(inputs, self.z_dim)
            sigma = tf.exp(self.z_log_sigma_sq / 2.)
            self.z = gaussian_sampler(self.z_mu, sigma)

            self.z_ph = tf.placeholder_with_default(tf.zeros_like(self.z_mu), shape=int_shape(self.z_mu))
            self.use_z_ph = tf.placeholder_with_default(False, shape=())

            use_z_ph = tf.cast(tf.cast(self.use_z_ph, tf.int32), tf.float32)
            z = self.z * (1-use_z_ph) + use_z_ph * self.z_ph

            if self.network_type == 'binary':
                self.pixel_params = decoder(z)
                self.x_hat = bernoulli_sampler(self.pixel_params, counters=self.counters)
            else:
                self.pixel_params = decoder(z, nr_logistic_mix=self.nr_logistic_mix)
                self.x_hat = mix_logistic_sampler(self.pixel_params, nr_logistic_mix=self.nr_logistic_mix, sample_range=self.sample_range, counters=self.counters)


    def __loss(self):
        print("******   Compute Loss   ******")
        if self.network_type == 'binary':
            self.loss_nll = bernoulli_loss(self.x, self.pixel_params, masks=self.masks, output_mean=False)
        else:
            self.loss_nll = mix_logistic_loss(self.x, self.pixel_params, masks=self.masks, output_mean=False)
        self.bits_per_dim = tf.reduce_mean(bits_per_dim_tf(nll=self.loss_nll, dim=tf.reduce_sum(1-self.masks, axis=[1,2])*self.num_channels))
        self.loss_nll = tf.reduce_mean(self.loss_nll)
        self.lam = 0.0
        if self.reg_type is None:
            self.loss_reg = 0
        elif self.reg_type =='kld':
            self.kld = compute_gaussian_kld(self.z_mu, self.z_log_sigma_sq)
            self.loss_reg = self.beta * tf.maximum(self.lam, self.kld)
        elif self.reg_type =='mmd':
            # self.mmd = estimate_mmd(tf.random_normal(int_shape(self.z)), self.z)
            self.mmd = estimate_mmd(tf.random_normal(tf.stack([256, self.z_dim])), self.z)
            self.loss_reg = self.beta * tf.maximum(self.lam, self.mmd)
        elif self.reg_type =='tc':
            self.mi, self.tc, self.dwkld = estimate_mi_tc_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=10000)
            self.loss_reg = self.mi + self.beta * self.tc + self.dwkld
        self.loss = self.loss_nll + self.loss_reg


@add_arg_scope
def conv_encoder_32_large_bn(inputs, z_dim, is_training=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_encoder_32_large_bn", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = inputs
            outputs = conv2d(outputs, 32, 1, 1, "SAME")
            outputs = conv2d(outputs, 64, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 256, 4, 2, "SAME")
            outputs = conv2d(outputs, 512, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 512])
            z_mu = dense(outputs, z_dim, nonlinearity=None, bn=True)
            z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=True)
            return z_mu, z_log_sigma_sq

@add_arg_scope
def conv_decoder_32_large_mixture_logistic(inputs, nr_logistic_mix, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conv_decoder_32_large_mixture_logistic", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = dense(inputs, 512)
            outputs = tf.reshape(outputs, [-1, 1, 1, 512])
            outputs = deconv2d(outputs, 256, 4, 1, "VALID")
            outputs = deconv2d(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d(outputs, 64, 4, 2, "SAME")
            outputs = deconv2d(outputs, 32, 4, 2, "SAME")
            outputs = deconv2d(outputs, 10 * nr_logistic_mix, 1, 1, "SAME", nonlinearity=None, bn=False)
            return outputs



@add_arg_scope
def conv_encoder_28_binary(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_encoder_28_binary", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = inputs
            outputs = conv2d(outputs, 32, 1, 1, "SAME")
            outputs = conv2d(outputs, 64, 4, 2, "SAME")
            outputs = conv2d(outputs, 64, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 1, "VALID")
            outputs = conv2d(outputs, 128, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 128])
            z_mu = dense(outputs, z_dim, nonlinearity=None, bn=True)
            z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=True)
            return z_mu, z_log_sigma_sq


@add_arg_scope
def conv_decoder_28_binary(inputs, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conv_decoder_28_binary", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = dense(inputs, 128)
            outputs = tf.reshape(outputs, [-1, 1, 1, 128])
            outputs = deconv2d(outputs, 128, 4, 1, "VALID")
            outputs = deconv2d(outputs, 64, 4, 1, "VALID")
            outputs = deconv2d(outputs, 64, 4, 2, "SAME")
            outputs = deconv2d(outputs, 32, 4, 2, "SAME")
            outputs = deconv2d(outputs, 1, 1, 1, "SAME", nonlinearity=None, bn=False)
            return outputs
