import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from blocks.helpers import Monitor, visualize_samples, get_nonlinearity, int_shape, get_trainable_variables
from blocks.optimizers import adam_updates
import data.load_data as load_data
from models.conv_vae import ConvVAE
from masks import get_generator
from configs import get_config

from learners.latent_model_learner import LatentModelLearner


parser = argparse.ArgumentParser()

config_path = sys.argv[1]
print(config_path)

with open(config_path) as config_file:
    cfg = json.load(config_file)

parser.add_argument('-is', '--img_size', type=int, default=cfg['img_size'], help="size of input image")
# data I/O
parser.add_argument('-sd', '--save_dir', type=str, default=cfg['save_dir'], help='Location for parameter checkpoints and samples')
parser.add_argument('-ds', '--data_set', type=str, default=cfg['data_set'], help='Can be either cifar|imagenet')
parser.add_argument('-si', '--save_interval', type=int, default=cfg['save_interval'], help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-lp', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
parser.add_argument('-bs', '--batch_size', type=int, default=cfg['batch_size'], help='Batch size during training per GPU')
parser.add_argument('-ng', '--nr_gpu', type=int, default=0, help='How many GPUs to distribute the training across?')
parser.add_argument('-g', '--gpus', type=str, default="", help='GPU No.s')
parser.add_argument('-rt', '--reg_type', type=str, default=cfg['reg_type'], help='Reg type')
parser.add_argument('-b', '--beta', type=int, default=cfg['beta'], help='Beta')
parser.add_argument('-zd', '--z_dim', type=int, default=cfg['z_dim'], help='z_dim')
parser.add_argument('-n', '--nonlinearity', type=str, default=cfg['nonlinearity'], help='')
parser.add_argument('-lr', '--learning_rate', type=float, default=cfg['learning_rate'], help='Base learning rate')
parser.add_argument('-mt', '--mask_type', type=str, default=cfg['mask_type'], help='mask type')
parser.add_argument('-nlm', '--nr_logistic_mix', type=float, default=cfg['nr_logistic_mix'], help="")
parser.add_argument('-sr', '--sample_range', type=float, default=cfg['sample_range'], help="")
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
# new features
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Under debug mode?')
parser.add_argument('-um', '--mode', type=str, default=cfg['mode'], help='')
parser.add_argument('-ns', '--network_type', type=str, default=cfg['network_type'], help='')
parser.add_argument('-en', '--exp_name', type=str, default=cfg['exp_name'], help='')

args = parser.parse_args(sys.argv[2:])
if args.mode in ['eval', 'test', 'generate', 'inpainting', 'traverse', 'inspect']:
    args.debug = True

args.nr_gpu = len(args.gpus.split(","))
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

tf.set_random_seed(args.seed)

model_opt = {
    "batch_size": args.batch_size,
    "img_size": args.img_size,
    "reg_type": args.reg_type,
    "beta": args.beta,
    "z_dim": args.z_dim,
    "nonlinearity": get_nonlinearity(args.nonlinearity),
    "bn": True,
    "kernel_initializer": tf.contrib.layers.xavier_initializer(),
    "kernel_regularizer": None,
    "nr_logistic_mix": args.nr_logistic_mix,
    "sample_range": args.sample_range,
    "network_type": args.network_type,
}

learner = LatentModelLearner(args.nr_gpu, args.save_dir, args.img_size, args.z_dim, exp_name=args.exp_name)
learner.load_data(args.data_set, args.batch_size*args.nr_gpu, use_debug_mode=args.debug)
learner.construct_models(model_cls=ConvVAE, model_opt=model_opt, learning_rate=args.learning_rate, eval_keys=['total loss', 'nll loss', 'reg loss', 'bits per dim'])
#
initializer = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(initializer)
    learner.set_session(sess)
    if args.mode == 'train':
        kwargs = {
            "train_mgen": get_generator(args.mask_type, size=args.img_size), ###
            "sample_mgen": get_generator(args.mask_type, size=args.img_size), ###
            "max_num_epoch": 200,
            "save_interval": args.save_interval,
            "restore": args.load_params,
        }
        learner.train(**kwargs)
    elif args.mode == 'test':
        learner.eval(which_set='test', mgen=get_generator(args.mask_type, size=args.img_size), generate_samples=False)
    elif args.mode == 'inpainting':
        layout = (10, 10)
        same_inputs = False
        use_mask_at = "{0}_{1}.npz".format(args.mask_type, args.data_set)
        learner.eval(which_set='test', mgen=get_generator(args.mask_type, size=args.img_size), generate_samples=True, layout=layout, same_inputs=same_inputs, use_mask_at=use_mask_at)
