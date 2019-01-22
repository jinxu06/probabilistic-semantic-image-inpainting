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

class Learner(object):

    def __init__(self, nr_gpu, save_dir, img_size, exp_name='default'):
        self.nr_gpu = nr_gpu
        self.save_dir = save_dir
        self.img_size = img_size
        self.exp_name = exp_name
        self.train_set = None
        self.eval_set = None
        self.test_set = None
        self.data_set = None
        self.batch_size = None
        self.monitor = None
        if not os.path.exists("results/"+self.exp_name):
            os.makedirs("results/"+self.exp_name)

    def load_data(self, dataset_name, batch_size, use_debug_mode=False):
        assert dataset_name in ['celeba', 'binarized-mnist', 'church_outdoor'], "cannot find the dataset"
        self.data_set = dataset_name
        self.batch_size = batch_size
        assert self.batch_size % self.nr_gpu == 0, "Batch of data cannot be evenly distributed to {0} GPUs".format(self.nr_gpu)
        if dataset_name == 'celeba':
            data_dir = "/data/ziz/not-backed-up/datasets-ziz-all/processed_data/CelebA"
            data_set = load_data.CelebA(data_dir=data_dir, batch_size=batch_size, img_size=self.img_size)
            self.num_channels = 3
            self.vrange = [-1., 1.]
        elif dataset_name == 'binarized-mnist':
            # data_dir = "/data/ziz/not-backed-up/datasets-ziz-all/processed_data/mnist"
            data_dir = "/data/ziz/not-backed-up/jxu/mnist"
            data_set = load_data.BinarizedMNIST(data_dir=data_dir, batch_size=batch_size, img_size=self.img_size)
            self.num_channels = 1
            self.vrange = [0, 1]
        elif dataset_name == 'church_outdoor':
            #data_dir = "/data/ziz/not-backed-up/datasets-ziz-all/raw_data/lsun/church_outdoor"
            data_dir = "/data/ziz/not-backed-up/jxu/church_outdoor"
            data_set = load_data.ChurchOutdoor(data_dir=data_dir, batch_size=batch_size, img_size=self.img_size)
            self.num_channels = 3
            self.vrange = [-1., 1.]

        if use_debug_mode:
            self.train_set = data_set.train(shuffle=True, limit=batch_size*2)
            self.eval_set = data_set.train(shuffle=True, limit=batch_size*2)
            self.test_set = data_set.test(shuffle=False, limit=-1)
        else:
            self.train_set = data_set.train(shuffle=True, limit=-1)
            self.eval_set = data_set.train(shuffle=True, limit=batch_size*10)
            self.test_set = data_set.test(shuffle=False, limit=-1)

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
                grads.append(tf.gradients(self.models[i].loss, self.params, colocate_gradients_with_ops=True))
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
        if 'mi' in eval_keys:
            mdict['mi'] = tf.add_n([model.mi for model in self.models]) / self.nr_gpu

        self.monitor = Monitor(dict=mdict, config_str="", log_file_path=self.save_dir+"/logfile")
        self.train_step = adam_updates(self.params, grads[0], lr=learning_rate)
        #
        self.saver = tf.train.Saver()

    def train_epoch(self, mgen, which_set='train'):
        raise NotImplementedError("Must override methodB")

    def eval_epoch(self, mgen, which_set='eval'):
        raise NotImplementedError("Must override")

    def sample(self, data, mgen):
        raise NotImplementedError("Must override")

    def preload(self, from_dir, var_list):
        preload_saver = tf.train.Saver(var_list=var_list)
        ckpt_file = from_dir + '/params_' + self.data_set + '.ckpt'
        print('restoring parameters from', ckpt_file)
        preload_saver.restore(self.sess, ckpt_file)

    def set_session(self, sess):
        self.sess = sess

    def save(self):
        self.saver.save(self.sess, self.save_dir + '/params_' + self.data_set + '.ckpt')

    def restore(self, saver=None, dir=None):
        if saver is None:
            saver = self.saver
        if dir is None:
            dir = self.save_dir
        ckpt_file = dir + '/params_' + self.data_set + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(self.sess, ckpt_file)


    def train(self, train_mgen, sample_mgen, max_num_epoch=100, save_interval=None, restore=False):
        if restore:
            self.restore()
        for epoch in range(max_num_epoch+1):
            tt = time.time()
            self.train_epoch(train_mgen, which_set='train')
            self.eval_epoch(train_mgen, which_set='eval')
            self.monitor.summarise_epoch(time=time.time()-tt, log=True)

            if save_interval is not None and epoch % save_interval == 0:
                self.save()
                data = next(self.test_set) # note that test set is used here
                self.test_set.reset()
                ori_x, masked_x, sample_x = self.sample(data, sample_mgen)
                visualize_samples(ori_x, os.path.join("results", self.exp_name, 'train_%s_gt_%d.png' % (self.data_set, epoch)), layout=(5, 5), vrange=self.vrange)
                visualize_samples(masked_x, os.path.join("results", self.exp_name, 'train_%s_masked_%d.png' % (self.data_set, epoch)), layout=(5, 5), vrange=self.vrange)
                visualize_samples(sample_x, os.path.join("results", self.exp_name, 'train_%s_sample_%d.png' % (self.data_set, epoch)), layout=(5, 5), vrange=self.vrange)
                print("------------ saved")
                sys.stdout.flush()

    def eval(self, which_set, mgen=None, generate_samples=False, restore=True, layout=(5,5), same_inputs=False, use_mask_at=None):
        if restore:
            self.restore()
        self.eval_epoch(mgen, which_set=which_set)
        self.monitor.summarise_epoch(time=0., log=False)
        if which_set == 'train':
            data_set = self.train_set
        elif which_set == 'eval':
            data_set = self.eval_set
        elif which_set == 'test':
            data_set = self.test_set
        if generate_samples:
            data = next(data_set)
            data_set.reset()
            ori_x, masked_x, sample_x = self.sample(data, mgen, same_inputs=same_inputs, use_mask_at=use_mask_at)
            visualize_samples(ori_x, os.path.join("results", self.exp_name, 'gen_%s_gt_%s.png' % (self.data_set, which_set)), layout=layout, vrange=self.vrange)
            visualize_samples(masked_x, os.path.join("results", self.exp_name, 'gen_%s_masked_%s.png' % (self.data_set, which_set)), layout=layout, vrange=self.vrange)
            visualize_samples(sample_x, os.path.join("results", self.exp_name, 'gen_%s_sample_%s.png' % (self.data_set, which_set)), layout=layout, vrange=self.vrange)
