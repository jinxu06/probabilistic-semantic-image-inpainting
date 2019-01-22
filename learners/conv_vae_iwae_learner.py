import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from blocks.helpers import Monitor, visualize_samples, get_nonlinearity, int_shape, get_trainable_variables, broadcast_masks_np
from blocks.optimizers import adam_updates
import data.load_data as load_data
from masks import get_generator
from .learner import Learner

class ConvVAEIWAELearner(Learner):

    def __init__(self, nr_gpu, save_dir, img_size, z_dim, exp_name="default"):
        super().__init__(nr_gpu, save_dir, img_size, exp_name)
        self.z_dim = z_dim


    def construct_models(self, model_cls, model_opt, learning_rate, trainable_params=None, eval_keys=['total loss']):
        # models
        self.models = [model_cls(counters={}) for i in range(self.nr_gpu)]
        template = tf.make_template('model', model_cls.build_graph)
        for i in range(self.nr_gpu):
            with tf.device('/gpu:%d' % i):
                template(self.models[i], **model_opt)
        if trainable_params is None:
            self.params = tf.trainable_variables()
        else:
            self.params = get_trainable_variables(trainable_params)
        # gradients
        grads = []
        for i in range(self.nr_gpu):
            with tf.device('/gpu:%d' % i):
                grad = self.models[i].gradients
                grads.append(grad)
        with tf.device('/gpu:0'):
            for i in range(1, self.nr_gpu):
                for j in range(len(grads[0])):
                    grads[0][j] += grads[i][j]

        mdict = {}
        if 'total loss' in eval_keys:
            mdict['total loss'] = tf.add_n([model.loss for model in self.models]) / self.nr_gpu
        if 'nll loss' in eval_keys:
            mdict['nll loss'] = tf.add_n([model.loss_nll for model in self.models]) / self.nr_gpu
        if 'reg loss' in eval_keys:
            mdict['reg loss'] = tf.add_n([model.loss_reg for model in self.models]) / self.nr_gpu
        if 'bits per dim' in eval_keys:
            mdict['bits per dim'] = tf.add_n([model.bits_per_dim for model in self.models]) / self.nr_gpu

        self.monitor = Monitor(dict=mdict, config_str="", log_file_path=self.save_dir+"/logfile")
        self.train_step = adam_updates(self.params, grads[0], lr=learning_rate)
        #
        self.saver = tf.train.Saver()


    def train_epoch(self, mgen, which_set='train'):
        if which_set == 'train':
            data_set = self.train_set
        elif which_set == 'eval':
            data_set = self.eval_set
        elif which_set == 'test':
            data_set = self.test_set
        for data in data_set:
            if self.num_channels == 3:
                data = np.cast[np.float32]((data - 127.5) / 127.5)
            ds = np.split(data, self.nr_gpu)
            feed_dict = {}
            feed_dict.update({model.is_training: True for model in self.models})
            feed_dict.update({model.dropout_p: 0.5 for model in self.models})
            feed_dict.update({model.x: ds[i] for i, model in enumerate(self.models)})
            feed_dict.update({model.x_bar: ds[i] for i, model in enumerate(self.models)})

            masks_np = [mgen.gen(self.batch_size//self.nr_gpu) for i in range(self.nr_gpu)]
            feed_dict.update({model.masks: masks_np[i] for i, model in enumerate(self.models)})
            feed_dict.update({model.input_masks: masks_np[i] for i, model in enumerate(self.models)})
            self.sess.run(self.train_step, feed_dict=feed_dict)

    def eval_epoch(self, mgen, which_set='eval'):
        if which_set == 'train':
            data_set = self.train_set
        elif which_set == 'eval':
            data_set = self.eval_set
        elif which_set == 'test':
            data_set = self.test_set
        for data in data_set:
            if self.num_channels == 3:
                data = np.cast[np.float32]((data - 127.5) / 127.5)
            ds = np.split(data, self.nr_gpu)
            feed_dict = {}
            feed_dict.update({model.is_training: False for model in self.models})
            feed_dict.update({model.dropout_p: 0.0 for model in self.models})
            feed_dict.update({model.x: ds[i] for i, model in enumerate(self.models)})
            feed_dict.update({model.x_bar: ds[i] for i, model in enumerate(self.models)})

            masks_np = [mgen.gen(self.batch_size//self.nr_gpu) for i in range(self.nr_gpu)]
            feed_dict.update({model.masks: masks_np[i] for i, model in enumerate(self.models)})
            feed_dict.update({model.input_masks: masks_np[i] for i, model in enumerate(self.models)})
            self.monitor.evaluate(self.sess, feed_dict)

    def sample(self, data, mgen):
        if self.num_channels == 3:
            data = np.cast[np.float32]((data - 127.5) / 127.5)
        ori_data = data.copy()
        ds = np.split(data.copy(), self.nr_gpu)
        feed_dict = {}
        feed_dict.update({model.is_training: False for model in self.models})
        feed_dict.update({model.dropout_p: 0.0 for model in self.models})
        feed_dict.update({model.x: ds[i] for i, model in enumerate(self.models)})
        feed_dict.update({model.x_bar: ds[i] for i, model in enumerate(self.models)})

        masks_np = [mgen.gen(self.batch_size//self.nr_gpu) for i in range(self.nr_gpu)]
        feed_dict.update({model.masks: masks_np[i] for i, model in enumerate(self.models)})
        feed_dict.update({model.input_masks: masks_np[i] for i, model in enumerate(self.models)})

        ret = self.sess.run([model.z_mu for model in self.models]+[model.z_log_sigma_sq for model in self.models], feed_dict=feed_dict)
        z_mu = np.concatenate(ret[:len(ret)//2], axis=0)
        z_log_sigma_sq = np.concatenate(ret[len(ret)//2:], axis=0)
        z_sigma = np.sqrt(np.exp(z_log_sigma_sq))
        z = np.random.normal(loc=z_mu, scale=z_sigma)
        z = np.split(z, self.nr_gpu)
        feed_dict.update({model.z_ph:z[i] for i, model in enumerate(self.models)})
        feed_dict.update({model.use_z_ph: True for model in self.models})

        for i in range(self.nr_gpu):
            ds[i] *= broadcast_masks_np(masks_np[i], num_channels=self.num_channels)
        masked_data = np.concatenate(ds, axis=0)
        x_gen = [ds[i].copy() for i in range(self.nr_gpu)]
        for yi in range(self.img_size):
            for xi in range(self.img_size):
                if np.min(np.array([masks_np[i][:, yi, xi] for i in range(self.nr_gpu)])) > 0:
                    continue
                feed_dict.update({model.x_bar:x_gen[i] for i, model in enumerate(self.models)})
                x_hats = self.sess.run([model.x_hat for model in self.models], feed_dict=feed_dict)
                for i in range(self.nr_gpu):
                    bmask = broadcast_masks_np(masks_np[i][:, yi, xi] , num_channels=self.num_channels)
                    x_gen[i][:, yi, xi, :] = x_hats[i][:, yi, xi, :] * (1.-bmask) + x_gen[i][:, yi, xi, :] * bmask
        gen_data = np.concatenate(x_gen, axis=0)
        return ori_data, masked_data, gen_data
