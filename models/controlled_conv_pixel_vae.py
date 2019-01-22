import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d, down_shift, right_shift
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shifted_deconv2d, up_left_shifted_deconv2d, up_shift, left_shift
from blocks.samplers import gaussian_sampler, mix_logistic_sampler, bernoulli_sampler
from blocks.estimators import compute_gaussian_kld, estimate_mmd, estimate_mi_tc_dwkld
from blocks.losses import mix_logistic_loss, bernoulli_loss
from blocks.helpers import int_shape, get_name, broadcast_masks_tf
from blocks.metrics import bits_per_dim_tf


class ControlledConvPixelVAE(object):

    def __init__(self, counters={}):
        self.counters = counters

    def build_graph(self, batch_size, img_size, z_dim, reg_type='kld', beta=1.0, network_type="large", sample_range=1.0, nonlinearity=tf.nn.elu, bn=True, kernel_initializer=None, kernel_regularizer=None, nr_resnet=1, nr_filters=100, nr_logistic_mix=10):

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
        self.nr_resnet = nr_resnet
        self.nr_filters = nr_filters
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
                decoder = conv_decoder_32_large
            else:
                encoder = conv_encoder_32
                decoder = conv_decoder_32
            forward_pixelcnn = forward_pixel_cnn_32_small
            reverse_pixelcnn = reverse_pixel_cnn_32_small
        elif self.img_size == 28:
            if self.network_type == 'binary':
                encoder = conv_encoder_28_binary
                decoder = conv_decoder_28_binary
                forward_pixelcnn = forward_pixel_cnn_28_binary
                reverse_pixelcnn = reverse_pixel_cnn_28_binary
        kwargs = {
            "nonlinearity": self.nonlinearity,
            "bn": self.bn,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "is_training": self.is_training,
            "counters": self.counters,
        }
        with arg_scope([forward_pixelcnn, reverse_pixelcnn, encoder, decoder], **kwargs):
            kwargs_pixelcnn = {
                "nr_resnet": self.nr_resnet,
                "nr_filters": self.nr_filters,
                "nr_logistic_mix": self.nr_logistic_mix,
                "dropout_p": self.dropout_p,
                "bn": False,
            }
            with arg_scope([forward_pixelcnn, reverse_pixelcnn], **kwargs_pixelcnn):
                inputs = self.x
                if self.input_masks is not None:
                    inputs = inputs * broadcast_masks_tf(self.input_masks, num_channels=self.num_channels)
                    inputs += tf.random_uniform(int_shape(inputs), -1, 1) * (1-broadcast_masks_tf(self.input_masks, num_channels=self.num_channels))
                    inputs = tf.concat([inputs, broadcast_masks_tf(self.input_masks, num_channels=1)], axis=-1)

                self.z_mu, self.z_log_sigma_sq = encoder(inputs, self.z_dim)
                sigma = tf.exp(self.z_log_sigma_sq / 2.)
                self.z = gaussian_sampler(self.z_mu, sigma)

                self.z_ph = tf.placeholder_with_default(tf.zeros_like(self.z_mu), shape=int_shape(self.z_mu))
                self.use_z_ph = tf.placeholder_with_default(False, shape=())

                use_z_ph = tf.cast(tf.cast(self.use_z_ph, tf.int32), tf.float32)
                z = self.z * (1-use_z_ph) + use_z_ph * self.z_ph

                decoded_features = decoder(z, output_features=True)
                # r_outputs = reverse_pixelcnn(self.x, self.masks, context=decoded_features, bn=False)
                r_outputs = reverse_pixelcnn(self.x, self.masks, context=None, bn=False)
                cond_features = tf.concat([r_outputs, decoded_features], axis=-1)
                cond_features = tf.concat([broadcast_masks_tf(self.input_masks, num_channels=1), cond_features], axis=-1)

                self.pixel_params = forward_pixelcnn(self.x_bar, cond_features, bn=False)

                if self.network_type == 'binary':
                    self.x_hat = bernoulli_sampler(self.pixel_params, counters=self.counters)
                else:
                    self.x_hat = mix_logistic_sampler(self.pixel_params, nr_logistic_mix=self.nr_logistic_mix, sample_range=self.sample_range, counters=self.counters)

    def __loss(self):
        print("******   Compute Loss   ******")
        if self.network_type == 'binary':
            self.loss_nll = bernoulli_loss(self.x, self.pixel_params, masks=self.masks, output_mean=False)
        else:
            self.loss_nll = mix_logistic_loss(self.x, self.pixel_params, masks=self.masks, output_mean=False)
        self.bits_per_dim = tf.reduce_mean(bits_per_dim_tf(nll=self.loss_nll, dim=tf.reduce_sum(1-self.masks, axis=[1,2])*self.num_channels))
        self.loss_nll = tf.reduce_mean(self.loss_nll)

        self.mi, self.tc, self.dwkld = estimate_mi_tc_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=10000)

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




    # def __loss(self, reg):
    #     print("******   Compute Loss   ******")
    #     self.mmd, self.kld, self.mi, self.tc, self.dwkld = [None for i in range(5)]
    #     self.gamma, self.dwmmd = 1e3, None ## hard coded, experimental
    #     self.mmdtc = None
    #     self.loss_ae = mix_logistic_loss(self.x, self.mix_logistic_params, masks=self.masks)
    #     if reg is None:
    #         self.loss_reg = 0
    #     elif reg=='kld':
    #         self.kld = compute_gaussian_kld(self.z_mu, self.z_log_sigma_sq)
    #         self.loss_reg = self.beta * tf.maximum(self.lam, self.kld)
    #     elif reg=='mmd':
    #         # self.mmd = estimate_mmd(tf.random_normal(int_shape(self.z)), self.z)
    #         self.mmd = estimate_mmd(tf.random_normal(tf.stack([256, self.z_dim])), self.z)
    #         self.loss_reg = self.beta * tf.maximum(self.lam, self.mmd)
    #     elif reg=='tc':
    #         self.mi, self.tc, self.dwkld = estimate_mi_tc_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=self.N)
    #         self.loss_reg = self.mi + self.beta * self.tc + self.dwkld
    #     elif reg=='info-tc':
    #         self.mi, self.tc, self.dwkld = estimate_mi_tc_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=self.N)
    #         self.loss_reg = self.beta * self.tc + self.dwkld
    #     elif reg=='tc-dwmmd':
    #         self.mi, self.tc, self.dwkld = estimate_mi_tc_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=self.N)
    #         self.dwmmd = estimate_mmd(tf.random_normal(int_shape(self.z)), self.z, is_dimention_wise=True)
    #         self.loss_reg = self.beta * self.tc + self.dwmmd * self.gamma
    #     elif reg=='mmd-tc':
    #         self.mmd = estimate_mmd(tf.random_normal(int_shape(self.z)), self.z)
    #         self.mmdtc = estimate_mmdtc(self.z, self.random_indices)
    #         self.loss_reg =  (self.mmd + self.beta * self.mmdtc) * 1e5
    #
    #     print("reg:{0}, beta:{1}, lam:{2}".format(self.reg, self.beta, self.lam))
    #     self.loss = self.loss_ae + self.loss_reg

@add_arg_scope
def conv_encoder_32(inputs, z_dim, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conv_encoder_32", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = inputs
            outputs = conv2d(outputs, 64, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 256, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 256])
            z_mu = dense(outputs, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=False)
            return z_mu, z_log_sigma_sq


@add_arg_scope
def conv_decoder_32(inputs, output_features=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conv_decoder_32", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = dense(inputs, 256)
            outputs = tf.reshape(outputs, [-1, 1, 1, 256])
            outputs = deconv2d(outputs, 128, 4, 1, "VALID")
            outputs = deconv2d(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d(outputs, 64, 4, 2, "SAME")
            if output_features:
                return deconv2d(outputs, 32, 4, 2, "SAME")
            outputs = deconv2d(outputs, 3, 4, 2, "SAME", nonlinearity=tf.sigmoid, bn=False)
            outputs = 2. * outputs - 1.
            return outputs


@add_arg_scope
def conv_encoder_32_large_bn(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
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
def conv_encoder_32_large(inputs, z_dim, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conv_encoder_32_large", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = inputs
            outputs = conv2d(outputs, 32, 1, 1, "SAME")
            outputs = conv2d(outputs, 32, 1, 1, "SAME")
            outputs = conv2d(outputs, 64, 4, 2, "SAME")
            outputs = conv2d(outputs, 128, 4, 2, "SAME")
            outputs = conv2d(outputs, 256, 4, 2, "SAME")
            outputs = conv2d(outputs, 512, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 512])
            z_mu = dense(outputs, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense(outputs, z_dim, nonlinearity=None, bn=False)
            return z_mu, z_log_sigma_sq


@add_arg_scope
def conv_decoder_32_large(inputs, output_features=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("conv_decoder_32_large", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = dense(inputs, 512)
            outputs = tf.reshape(outputs, [-1, 1, 1, 512])
            outputs = deconv2d(outputs, 256, 4, 1, "VALID")
            outputs = deconv2d(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d(outputs, 64, 4, 2, "SAME")
            outputs = deconv2d(outputs, 32, 4, 2, "SAME")
            if output_features:
                return outputs
            outputs = deconv2d(outputs, 3, 1, 1, "SAME", nonlinearity=tf.sigmoid, bn=False)
            outputs = 2. * outputs - 1.
            return outputs




@add_arg_scope
def forward_pixel_cnn_32_small(x, context, nr_logistic_mix=10, nr_resnet=1, nr_filters=100, dropout_p=0.0, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("forward_pixel_cnn_32_small", counters)
    print("construct", name, "...")
    print("    * nr_resnet: ", nr_resnet)
    print("    * nr_filters: ", nr_filters)
    print("    * nr_logistic_mix: ", nr_logistic_mix)
    assert not bn, "auto-reggressive model should not use batch normalization"
    with tf.variable_scope(name):
        with arg_scope([gated_resnet], gh=None, sh=context, nonlinearity=nonlinearity, dropout_p=dropout_p):
            with arg_scope([gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
                xs = int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                x_out = nin(tf.nn.elu(ul_list[-1]),10*nr_logistic_mix)
                return x_out

@add_arg_scope
def reverse_pixel_cnn_32_small(x, masks, context=None, nr_logistic_mix=10, nr_resnet=1, nr_filters=100, dropout_p=0.0, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("reverse_pixel_cnn_32_small", counters)
    x = x * broadcast_masks_tf(masks, num_channels=3)
    x = tf.concat([x, broadcast_masks_tf(masks, num_channels=1)], axis=-1)
    print("construct", name, "...")
    print("    * nr_resnet: ", nr_resnet)
    print("    * nr_filters: ", nr_filters)
    print("    * nr_logistic_mix: ", nr_logistic_mix)
    assert not bn, "auto-reggressive model should not use batch normalization"
    with tf.variable_scope(name):
        with arg_scope([gated_resnet], gh=None, sh=context, nonlinearity=nonlinearity, dropout_p=dropout_p):
            with arg_scope([gated_resnet, up_shifted_conv2d, up_left_shifted_conv2d, up_shifted_deconv2d, up_left_shifted_deconv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
                xs = int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [up_shift(up_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [up_shift(up_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        left_shift(up_left_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=up_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=up_left_shifted_conv2d))

                x_out = nin(tf.nn.elu(ul_list[-1]), nr_filters)
                return x_out



########################


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
def conv_decoder_28_binary(inputs, output_features=False, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
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
            if output_features:
                return outputs
            outputs = deconv2d(outputs, 1, 1, 1, "SAME", nonlinearity=tf.sigmoid, bn=False)
            outputs = 2. * outputs - 1.
            return outputs


@add_arg_scope
def forward_pixel_cnn_28_binary(x, context, nr_logistic_mix=10, nr_resnet=1, nr_filters=100, dropout_p=0.0, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("forward_pixel_cnn_28_binary", counters)
    print("construct", name, "...")
    print("    * nr_resnet: ", nr_resnet)
    print("    * nr_filters: ", nr_filters)
    print("    * nr_logistic_mix: ", nr_logistic_mix)
    assert not bn, "auto-reggressive model should not use batch normalization"
    with tf.variable_scope(name):
        with arg_scope([gated_resnet], gh=None, sh=context, nonlinearity=nonlinearity, dropout_p=dropout_p):
            with arg_scope([gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
                xs = int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                x_out = nin(tf.nn.elu(ul_list[-1]), 1)
                return x_out

@add_arg_scope
def reverse_pixel_cnn_28_binary(x, masks, context=None, nr_logistic_mix=10, nr_resnet=1, nr_filters=100, dropout_p=0.0, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("reverse_pixel_cnn_28_binary", counters)
    x = x * broadcast_masks_tf(masks, num_channels=3)
    x = tf.concat([x, broadcast_masks_tf(masks, num_channels=1)], axis=-1)
    print("construct", name, "...")
    print("    * nr_resnet: ", nr_resnet)
    print("    * nr_filters: ", nr_filters)
    print("    * nr_logistic_mix: ", nr_logistic_mix)
    assert not bn, "auto-reggressive model should not use batch normalization"
    with tf.variable_scope(name):
        with arg_scope([gated_resnet], gh=None, sh=context, nonlinearity=nonlinearity, dropout_p=dropout_p):
            with arg_scope([gated_resnet, up_shifted_conv2d, up_left_shifted_conv2d, up_shifted_deconv2d, up_left_shifted_deconv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
                xs = int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [up_shift(up_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [up_shift(up_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        left_shift(up_left_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=up_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=up_left_shifted_conv2d))

                x_out = nin(tf.nn.elu(ul_list[-1]), nr_filters)
                return x_out
