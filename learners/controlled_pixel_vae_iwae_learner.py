import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from blocks.helpers import Monitor
from blocks.helpers import visualize_samples, get_nonlinearity, int_shape, get_trainable_variables, broadcast_masks_np
from blocks.optimizers import adam_updates
import data.load_data as load_data
from masks import get_generator
from .latent_model_learner import LatentModelLearner

class ControlledPixelVAEIWAELearner(LatentModelLearner):

    def __init__(self, nr_gpu, save_dir, img_size, z_dim, phase, exp_name="default"):
        super().__init__(nr_gpu, save_dir, img_size, z_dim, exp_name=exp_name)
        self.phase = phase


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
            if self.phase == 'pvae':
                feed_dict.update({model.masks: np.zeros_like(masks_np[i]) for i, model in enumerate(self.models)})
            elif self.phase == 'ce':
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
            if self.phase == 'pvae':
                feed_dict.update({model.masks: np.zeros_like(masks_np[i]) for i, model in enumerate(self.models)})
            elif self.phase == 'ce':
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
        if self.phase == 'pvae':
            feed_dict.update({model.masks: np.zeros_like(masks_np[i]) for i, model in enumerate(self.models)})
        elif self.phase == 'ce':
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

    def latent_traversal(self, context, traversal_range=[-6, 6], num_traversal_step=13, mgen=None):
        self.num_traversal_step = num_traversal_step
        if self.num_channels == 3:
            image = np.cast[np.float32]((context - 127.5) / 127.5)
        num_instances = num_traversal_step * self.z_dim
        assert num_instances <= self.batch_size, "cannot feed all the instances into GPUs"
        data = np.stack([image.copy() for i in range(self.batch_size)], axis=0)
        ori_data = data.copy()
        ds = np.split(data.copy(), self.nr_gpu)

        feed_dict = {}
        feed_dict.update({model.is_training: False for model in self.models})
        feed_dict.update({model.dropout_p: 0.0 for model in self.models})
        feed_dict.update({model.x: ds[i] for i, model in enumerate(self.models)})

        masks_np = [mgen.gen(self.batch_size//self.nr_gpu) for i in range(self.nr_gpu)]
        if self.phase == 'pvae':
            feed_dict.update({model.masks: np.zeros_like(masks_np[i]) for i, model in enumerate(self.models)})
        elif self.phase == 'ce':
            feed_dict.update({model.masks: masks_np[i] for i, model in enumerate(self.models)})
        feed_dict.update({model.input_masks: masks_np[i] for i, model in enumerate(self.models)})

        ret = self.sess.run([model.z_mu for model in self.models]+[model.z_log_sigma_sq for model in self.models], feed_dict=feed_dict)
        z_mu = np.concatenate(ret[:len(ret)//2], axis=0)
        z_log_sigma_sq = np.concatenate(ret[len(ret)//2:], axis=0)
        z_sigma = np.sqrt(np.exp(z_log_sigma_sq))
        z = z_mu #np.random.normal(loc=z_mu, scale=z_sigma)
        for i in range(z.shape[0]):
            z[i] = z[0].copy()
        for i in range(self.z_dim):
            z[i*num_traversal_step:(i+1)*num_traversal_step, i] = np.linspace(start=traversal_range[0], stop=traversal_range[1], num=num_traversal_step)
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
        return ori_data[:num_instances], masked_data[:num_instances], gen_data[:num_instances]

    def latent_inspect(self, image, traversal_range=[-6, 6], num_traversal_step=13):
        self.num_traversal_step = num_traversal_step
        if self.num_channels == 3:
            image = np.cast[np.float32]((image - 127.5) / 127.5)
        num_instances = num_traversal_step * self.z_dim
        assert num_instances <= self.batch_size, "cannot feed all the instances into GPUs"
        data = np.stack([image.copy() for i in range(self.batch_size)], axis=0)
        ori_data = data.copy()
        ds = np.split(data.copy(), self.nr_gpu)

        feed_dict = {}
        feed_dict.update({model.is_training: False for model in self.models})
        feed_dict.update({model.dropout_p: 0.0 for model in self.models})
        feed_dict.update({model.x: ds[i] for i, model in enumerate(self.models)})

        feed_dict.update({model.masks: np.zeros((self.batch_size//self.nr_gpu, self.img_size, self.img_size)) for i, model in enumerate(self.models)})
        feed_dict.update({model.input_masks: np.ones((self.batch_size//self.nr_gpu, self.img_size, self.img_size)) for i, model in enumerate(self.models)})

        ret = self.sess.run([model.z_mu for model in self.models]+[model.z_log_sigma_sq for model in self.models], feed_dict=feed_dict)
        z_mu = np.concatenate(ret[:len(ret)//2], axis=0)
        z_log_sigma_sq = np.concatenate(ret[len(ret)//2:], axis=0)
        z_sigma = np.sqrt(np.exp(z_log_sigma_sq))

        z = z_mu #np.random.normal(loc=z_mu, scale=z_sigma)
        for i in range(z.shape[0]):
            z[i] = z[0].copy()
        for i in range(self.z_dim):
            z[i*num_traversal_step:(i+1)*num_traversal_step, i] = np.linspace(start=traversal_range[0], stop=traversal_range[1], num=num_traversal_step)
        z = np.split(z, self.nr_gpu)

        feed_dict.update({model.z_ph:z[i] for i, model in enumerate(self.models)})
        feed_dict.update({model.use_z_ph: True for model in self.models})

        x_gen = [ds[i].copy() for i in range(self.nr_gpu)]
        for yi in range(self.img_size):
            for xi in range(self.img_size):
                feed_dict.update({model.x_bar:x_gen[i] for i, model in enumerate(self.models)})
                x_hats = self.sess.run([model.x_hat for model in self.models], feed_dict=feed_dict)
                for i in range(self.nr_gpu):
                    x_gen[i][:, yi, xi, :] = x_hats[i][:, yi, xi, :]
        gen_data = np.concatenate(x_gen, axis=0)
        return gen_data[:num_instances]

    def inpainting(self, eval_mgen):
        self.restore()
        #
        data = next(self.test_set)
        self.test_set.reset()
        ori_x, masked_x, sample_x = self.sample(data, eval_mgen)
        visualize_samples(ori_x, os.path.join("results", self.exp_name, 'inpainting_%s_gt.png' % (self.data_set)), layout=(5, 5), vrange=self.vrange)
        visualize_samples(masked_x, os.path.join("results", self.exp_name, 'inpainting_%s_masked_%s.png' % (self.data_set)), layout=(5, 5), vrange=self.vrange)
        visualize_samples(sample_x, os.path.join("results", self.exp_name, 'inpainting_%s_sample_%s.png' % (self.data_set)), layout=(5, 5), vrange=self.vrange)

    def traverse(self, eval_mgen, image_id=0):
        self.restore()
        data = next(self.test_set, image_id+1)[image_id]
        self.test_set.reset()
        ori_x, masked_x, sample_x = self.latent_traversal(data, mgen=eval_mgen)
        visualize_samples(ori_x, os.path.join("results",'traverse_%s_gt_%d_%s.png' % (self.data_set, image_id, eval_mgen.name)), layout=(self.z_dim, self.num_traversal_step), vrange=self.vrange)
        visualize_samples(masked_x, os.path.join("results",'traverse_%s_masked_%d_%s.png' % (self.data_set, image_id, eval_mgen.name)), layout=(self.z_dim, self.num_traversal_step), vrange=self.vrange)
        visualize_samples(sample_x, os.path.join("results",'traverse_%s_sample_%d_%s.png' % (self.data_set, image_id, eval_mgen.name)), layout=(self.z_dim, self.num_traversal_step), vrange=self.vrange)

    def inspect(self, image_id=0):
        self.restore()
        data = next(self.test_set, image_id+1)[image_id]
        self.test_set.reset()
        sample_x = self.latent_inspect(data)
        visualize_samples(sample_x, os.path.join("results",'inspect_%s_sample_%d.png' % (self.data_set, image_id)), layout=(self.z_dim, self.num_traversal_step), vrange=self.vrange)
