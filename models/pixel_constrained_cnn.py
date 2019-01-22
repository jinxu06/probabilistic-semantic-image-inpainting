import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense, nin, gated_resnet
from blocks.layers import up_shifted_conv2d, up_left_shifted_conv2d, up_shift, left_shift
from blocks.layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d, down_shift, right_shift
from blocks.samplers import gaussian_sampler, mix_logistic_sampler, softmax_sampler, bernoulli_sampler
from blocks.estimators import compute_gaussian_kld
from blocks.losses import mix_logistic_loss, bernoulli_loss
from blocks.helpers import int_shape, get_name, broadcast_masks_tf, params_to_logits
from blocks.layers import residual_block
from blocks.metrics import bits_per_dim_tf



class PixelConstrainedCNN(object):

    def __init__(self, counters={}):
        self.counters = counters

    def build_graph(self, batch_size, img_size, network_type="large", sample_range=1.0, nonlinearity=tf.nn.elu, bn=True, kernel_initializer=None, kernel_regularizer=None, nr_resnet=1, nr_filters=100, nr_logistic_mix=10):

        self.batch_size = batch_size
        self.img_size = img_size
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
        # choose network size
        if self.img_size == 32:
            if self.network_type == 'large':
                conditioning_network = conditioning_network_32
                prior_network = forward_pixel_cnn_32
            else:
                raise Exception("unknown network type")
        elif self.img_size == 28:
            if self.network_type == 'binary':
                conditioning_network = conditioning_network_28
                prior_network = forward_pixel_cnn_28_binary
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
        with arg_scope([conditioning_network, prior_network], **kwargs):
            kwargs_pixelcnn = {
                "nr_resnet": self.nr_resnet,
                "nr_filters": self.nr_filters,
                "nr_logistic_mix": self.nr_logistic_mix,
                "dropout_p": self.dropout_p,
                "bn": False,
            }
            with arg_scope([prior_network], **kwargs_pixelcnn):
                self.cond_features = conditioning_network(self.x, self.masks, nr_filters=self.nr_filters)
                self.pixel_params = prior_network(self.x_bar, self.cond_features, bn=False)
                if self.network_type == 'binary':
                    self.x_hat = bernoulli_sampler(self.pixel_params, counters=self.counters)
                else:
                    self.x_hat = mix_logistic_sampler(self.pixel_params, nr_logistic_mix=self.nr_logistic_mix, sample_range=self.sample_range, counters=self.counters)


    def __loss(self):
        print("******   Compute Loss   ******")
        if self.network_type == 'binary':
            self.loss = bernoulli_loss(self.x, self.pixel_params, masks=self.masks, output_mean=False)
        else:
            self.loss = mix_logistic_loss(self.x, self.pixel_params, masks=self.masks, output_mean=False)
        self.bits_per_dim = tf.reduce_mean(bits_per_dim_tf(nll=self.loss, dim=tf.reduce_sum(1-self.masks, axis=[1,2])*self.num_channels))
        self.loss = tf.reduce_mean(self.loss)
        self.loss_nll = self.loss

@add_arg_scope
def conditioning_network_32(x, masks, nr_filters, is_training=True, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conditioning_network_32", counters)
    x = x * broadcast_masks_tf(masks, num_channels=3)
    x = tf.concat([x, broadcast_masks_tf(masks, num_channels=1)], axis=-1)
    xs = int_shape(x)
    x = tf.concat([x,tf.ones(xs[:-1]+[1])],3)
    with tf.variable_scope(name):
        with arg_scope([conv2d, residual_block, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = conv2d(x, nr_filters, 4, 1, "SAME")
            for l in range(4):
                # outputs = residual_block(outputs, nr_filters, 5, 1, "SAME")
                outputs = conv2d(outputs, nr_filters, 4, 1, "SAME")
            outputs = conv2d(outputs, nr_filters, 1, 1, "SAME", nonlinearity=None, bn=False)
            return outputs

@add_arg_scope
def conditioning_network_28(x, masks, nr_filters, is_training=True, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conditioning_network_28", counters)
    x = x * broadcast_masks_tf(masks, num_channels=3)
    x = tf.concat([x, broadcast_masks_tf(masks, num_channels=1)], axis=-1)
    xs = int_shape(x)
    x = tf.concat([x,tf.ones(xs[:-1]+[1])],3)
    with tf.variable_scope(name):
        with arg_scope([conv2d, residual_block, dense], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
            outputs = conv2d(x, nr_filters, 4, 1, "SAME")
            for l in range(4):
                outputs = conv2d(outputs, nr_filters, 4, 1, "SAME")
            outputs = conv2d(outputs, nr_filters, 1, 1, "SAME", nonlinearity=None, bn=False)
            return outputs


@add_arg_scope
def forward_pixel_cnn_32(x, context, nr_logistic_mix=10, nr_resnet=1, nr_filters=100, dropout_p=0.0, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("forward_pixel_cnn_32", counters)
    print("construct", name, "...")
    print("    * nr_resnet: ", nr_resnet)
    print("    * nr_filters: ", nr_filters)
    print("    * nr_logistic_mix: ", nr_logistic_mix)
    assert not bn, "auto-reggressive model should not use batch normalization"
    with tf.variable_scope(name):
        with arg_scope([gated_resnet], gh=None, sh=None, nonlinearity=nonlinearity, dropout_p=dropout_p):
            with arg_scope([gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
                xs = int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], sh=context, conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], sh=context, conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(u_list[-1], num_filters=nr_filters, strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(u_list[-1], num_filters=nr_filters, strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                # /////// down pass ////////

                u = u_list.pop()
                ul = ul_list.pop()

                for rep in range(nr_resnet):
                    u = gated_resnet(u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(u, num_filters=nr_filters, strides=[2, 2])
                ul = down_right_shifted_deconv2d(ul, num_filters=nr_filters, strides=[2, 2])

                for rep in range(nr_resnet+1):
                    u = gated_resnet(u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(u, num_filters=nr_filters, strides=[2, 2])
                ul = down_right_shifted_deconv2d(ul, num_filters=nr_filters, strides=[2, 2])


                for rep in range(nr_resnet+1):
                    u = gated_resnet(u, u_list.pop(), sh=context, conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat([u, ul_list.pop()],3), sh=context, conv=down_right_shifted_conv2d)

                x_out = nin(tf.nn.elu(ul),10*nr_logistic_mix)
                assert len(u_list) == 0
                assert len(ul_list) == 0
                return x_out




@add_arg_scope
def forward_pixel_cnn_28_binary(x, context, nr_logistic_mix=10, nr_resnet=1, nr_filters=100, dropout_p=0.0, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = get_name("forward_pixel_cnn_28", counters)
    print("construct", name, "...")
    print("    * nr_resnet: ", nr_resnet)
    print("    * nr_filters: ", nr_filters)
    print("    * nr_logistic_mix: ", nr_logistic_mix)
    assert not bn, "auto-reggressive model should not use batch normalization"
    with tf.variable_scope(name):
        with arg_scope([gated_resnet], gh=None, sh=None, nonlinearity=nonlinearity, dropout_p=dropout_p):
            with arg_scope([gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d], bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training, counters=counters):
                xs = int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], sh=context, conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], sh=context, conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(u_list[-1], num_filters=nr_filters, strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                u_list.append(down_shifted_conv2d(u_list[-1], num_filters=nr_filters, strides=[2, 2]))
                ul_list.append(down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, strides=[2, 2]))

                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

                # /////// down pass ////////

                u = u_list.pop()
                ul = ul_list.pop()

                for rep in range(nr_resnet):
                    u = gated_resnet(u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(u, num_filters=nr_filters, strides=[2, 2])
                ul = down_right_shifted_deconv2d(ul, num_filters=nr_filters, strides=[2, 2])

                for rep in range(nr_resnet+1):
                    u = gated_resnet(u, u_list.pop(), conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=down_right_shifted_conv2d)

                u = down_shifted_deconv2d(u, num_filters=nr_filters, strides=[2, 2])
                ul = down_right_shifted_deconv2d(ul, num_filters=nr_filters, strides=[2, 2])


                for rep in range(nr_resnet+1):
                    u = gated_resnet(u, u_list.pop(), sh=None, conv=down_shifted_conv2d)
                    ul = gated_resnet(ul, tf.concat([u, ul_list.pop()],3), sh=None, conv=down_right_shifted_conv2d)

                x_out = nin(tf.nn.elu(ul), 1)
                assert len(u_list) == 0
                assert len(ul_list) == 0
                return x_out
