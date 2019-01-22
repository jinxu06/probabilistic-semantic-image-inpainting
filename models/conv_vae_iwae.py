import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d, down_shift, right_shift
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shifted_deconv2d, up_left_shifted_deconv2d, up_shift, left_shift
from blocks.samplers import gaussian_sampler, mix_logistic_sampler, bernoulli_sampler
from blocks.estimators import compute_gaussian_kld, estimate_mmd, compute_2gaussian_kld
from blocks.losses import mix_logistic_loss, bernoulli_loss
from blocks.helpers import int_shape, get_name, broadcast_masks_tf, get_trainable_variables
from blocks.metrics import bits_per_dim_tf

class ConvVAEIWAE(object):

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
                encoder_q = conv_encoder_32_q
            else:
                raise Exception("unknown network type")
        elif self.img_size == 28:
            if self.network_type == 'binary':
                encoder = conv_encoder_28_binary
                decoder = conv_decoder_28_binary
                encoder_q = conv_encoder_28_binary_q
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
        with arg_scope([encoder, decoder, encoder_q], **kwargs):
            inputs = self.x

            self.num_particles = 16

            inputs = inputs * broadcast_masks_tf(self.masks, num_channels=self.num_channels)
            inputs = tf.concat([inputs, broadcast_masks_tf(self.masks, num_channels=1)], axis=-1)
            inputs_pos = tf.concat([self.x, broadcast_masks_tf(self.masks, num_channels=1)], axis=-1)
            inputs = tf.concat([inputs, inputs_pos], axis=0)

            z_mu, z_log_sigma_sq = encoder(inputs, self.z_dim)
            self.z_mu_pr, self.z_mu = z_mu[:self.batch_size], z_mu[self.batch_size:]
            self.z_log_sigma_sq_pr, self.z_log_sigma_sq = z_log_sigma_sq[:self.batch_size], z_log_sigma_sq[self.batch_size:]

            self.z_mu, self.z_log_sigma_sq = self.z_mu_pr, self.z_log_sigma_sq_pr

            x = tf.tile(self.x, [self.num_particles, 1, 1, 1])
            masks = tf.tile(self.masks, [self.num_particles, 1, 1])
            self.z_mu = tf.tile(self.z_mu, [self.num_particles, 1])
            self.z_mu_pr = tf.tile(self.z_mu_pr, [self.num_particles, 1])
            self.z_log_sigma_sq = tf.tile(self.z_log_sigma_sq, [self.num_particles, 1])
            self.z_log_sigma_sq_pr = tf.tile(self.z_log_sigma_sq_pr, [self.num_particles, 1])
            sigma = tf.exp(self.z_log_sigma_sq / 2.)

            self.params = get_trainable_variables(["inference"])

            dist = tf.distributions.Normal(loc=0., scale=1.)
            epsilon = dist.sample(sample_shape=[self.batch_size*self.num_particles, self.z_dim], seed=None)
            z = self.z_mu + tf.multiply(epsilon, sigma)

            if self.network_type == 'binary':
                self.pixel_params = decoder(z)
            else:
                self.pixel_params = decoder(z, nr_logistic_mix=self.nr_logistic_mix)
            if self.network_type == 'binary':
                nll = bernoulli_loss(x, self.pixel_params, masks=masks, output_mean=False)
            else:
                nll = mix_logistic_loss(x, self.pixel_params, masks=masks, output_mean=False)

            log_prob_pos = dist.log_prob(epsilon)
            epsilon_pr = (z - self.z_mu_pr) / tf.exp(self.z_log_sigma_sq_pr / 2.)
            log_prob_pr = dist.log_prob(epsilon_pr)
            # convert back
            log_prob_pr = tf.stack([log_prob_pr[self.batch_size*i:self.batch_size*(i+1)] for i in range(self.num_particles)], axis=0)
            log_prob_pos = tf.stack([log_prob_pos[self.batch_size*i:self.batch_size*(i+1)] for i in range(self.num_particles)], axis=0)
            log_prob_pr = tf.reduce_sum(log_prob_pr, axis=2)
            log_prob_pos = tf.reduce_sum(log_prob_pos, axis=2)
            nll = tf.stack([nll[self.batch_size*i:self.batch_size*(i+1)] for i in range(self.num_particles)], axis=0)
            log_likelihood = - nll

            # log_weights = log_prob_pr + log_likelihood - log_prob_pos
            log_weights = log_likelihood
            log_sum_weight = tf.reduce_logsumexp(log_weights, axis=0)
            log_avg_weight = log_sum_weight - tf.log(tf.to_float(self.num_particles))
            self.log_avg_weight = log_avg_weight

            normalized_weights = tf.stop_gradient(tf.nn.softmax(log_weights, axis=0))
            sq_normalized_weights = tf.square(normalized_weights)

            self.gradients = tf.gradients(-tf.reduce_sum(sq_normalized_weights * log_weights, axis=0), self.params, colocate_gradients_with_ops=True)

    def __loss(self):
        print("******   Compute Loss   ******")

        self.loss = - self.log_avg_weight
        self.bits_per_dim = tf.reduce_mean(bits_per_dim_tf(nll=self.loss, dim=tf.reduce_sum(1-self.masks, axis=[1,2])*self.num_channels))
        self.loss = tf.reduce_mean(self.loss)

        # if self.network_type == 'binary':
        #     self.loss_nll = bernoulli_loss(self.x, self.pixel_params, masks=self.masks, output_mean=False)
        # else:
        #     self.loss_nll = mix_logistic_loss(self.x, self.pixel_params, masks=self.masks, output_mean=False)
        # self.bits_per_dim = tf.reduce_mean(bits_per_dim_tf(nll=self.loss_nll, dim=tf.reduce_sum(1-self.masks, axis=[1,2])*self.num_channels))
        # self.loss_nll = tf.reduce_mean(self.loss_nll)
        #
        # self.kld = compute_2gaussian_kld(self.z_mu, self.z_log_sigma_sq, self.z_mu_pr, self.z_log_sigma_sq_pr)
        # self.loss_reg = self.kld
        #
        # self.loss = self.loss_nll + self.loss_reg

        # self.num_particles = 5
        # dist = tf.distributions.Normal(loc=0., scale=1.)
        # epsilon = dist.sample(sample_shape=[self.batch_size, self.z_dim, self.num_particles], seed=None)
        #
        # sigma = tf.exp(self.z_log_sigma_sq / 2.)
        # z = self.z_mu + tf.multiply(epsilon, sigma)
        #
        # log_prob_pos = tf.reduce_sum(dist.log_prob(epsilon), axis=1)
        # epsilon_pr = (z - self.z_mu_pr) / tf.exp(self.z_log_sigma_sq_pr / 2.)
        # log_prob_pr = tf.reduce_sum(dist.log_prob(epsilon_pr), axis=1)
        #
        # w = log_prob_pr / log_prob_pos
        # print(w)
        # quit()

        # self.params = get_trainable_variables(["inference"])
        # self.gradients = tf.gradients(self.loss, self.params, colocate_gradients_with_ops=True)


@add_arg_scope
def conv_encoder_32_q(inputs, z_dim, is_training=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("inference_32", counters)
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
def conv_encoder_28_binary_q(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("inference_28", counters)
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
