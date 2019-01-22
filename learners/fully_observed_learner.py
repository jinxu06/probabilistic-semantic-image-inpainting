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
from .learner import Learner

class FullyObservedLearner(Learner):

    def __init__(self, nr_gpu, save_dir, img_size, exp_name="default"):
        super().__init__(nr_gpu, save_dir, img_size, exp_name)

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
            self.monitor.evaluate(self.sess, feed_dict)

    def sample(self, data, mgen, same_inputs=False, use_mask_at=None):
        if self.num_channels == 3:
            data = np.cast[np.float32]((data - 127.5) / 127.5)
        if same_inputs:
            for i in range(data.shape[0]):
                data[i] = data[3]
        ori_data = data.copy()
        ds = np.split(data.copy(), self.nr_gpu)
        feed_dict = {}
        feed_dict.update({model.is_training: False for model in self.models})
        feed_dict.update({model.dropout_p: 0.0 for model in self.models})
        feed_dict.update({model.x: ds[i] for i, model in enumerate(self.models)})
        feed_dict.update({model.x_bar: ds[i] for i, model in enumerate(self.models)})

        if use_mask_at is not None:
            masks_np = np.load(use_mask_at)['masks']
            masks_np = np.split(masks_np, self.nr_gpu)
        else:
            masks_np = [mgen.gen(self.batch_size//self.nr_gpu) for i in range(self.nr_gpu)]
            np.savez(mgen.name+"_"+self.data_set, masks=np.concatenate(masks_np))

        if same_inputs:
            for g in range(self.nr_gpu):
                for i in range(self.batch_size//self.nr_gpu):
                    masks_np[g][i] = masks_np[0][0]
        feed_dict.update({model.masks: masks_np[i] for i, model in enumerate(self.models)})
        #
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
        if self.num_channels == 1:
            masks_np = np.concatenate(masks_np, axis=0)
            masks_np = broadcast_masks_np(masks_np, num_channels=self.num_channels)
            masked_data += (1-masks_np) * 0.5
        return ori_data, masked_data, gen_data
