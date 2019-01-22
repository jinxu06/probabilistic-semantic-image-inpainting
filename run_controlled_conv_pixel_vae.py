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
from models.controlled_conv_pixel_vae import ControlledConvPixelVAE
from masks import get_generator
from configs import get_config
from learners.controlled_pixel_vae_learner import ControlledPixelVAELearner


parser = argparse.ArgumentParser()

config_path = sys.argv[1]
print(config_path)

with open(config_path) as config_file:
    cfg = json.load(config_file)

parser.add_argument('-is', '--img_size', type=int, default=cfg['img_size'], help="size of input image")
# data I/O
parser.add_argument('-sd', '--save_dir', type=str, default=cfg['save_dir'], help='Location for parameter checkpoints and samples')
parser.add_argument('-ds', '--data_set', type=str, default=cfg['data_set'], help='Can be either cifar|imagenet')
parser.add_argument('-p', '--phase', type=str, default=cfg['phase'], help='')
parser.add_argument('-si', '--save_interval', type=int, default=cfg['save_interval'], help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-lp', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
parser.add_argument('-bs', '--batch_size', type=int, default=cfg['batch_size'], help='Batch size during training per GPU')
parser.add_argument('-ng', '--nr_gpu', type=int, default=0, help='How many GPUs to distribute the training across?')
parser.add_argument('-g', '--gpus', type=str, default="", help='GPU No.s')
parser.add_argument('-rt', '--reg_type', type=str, default=cfg['reg_type'], help='Reg type')
parser.add_argument('-mt', '--mask_type', type=str, default=cfg['mask_type'], help='mask type')
parser.add_argument('-b', '--beta', type=int, default=cfg['beta'], help='Beta')
parser.add_argument('-zd', '--z_dim', type=int, default=cfg['z_dim'], help='z_dim')
parser.add_argument('-n', '--nonlinearity', type=str, default=cfg['nonlinearity'], help='')
parser.add_argument('-lr', '--learning_rate', type=float, default=cfg['learning_rate'], help='Base learning rate')
parser.add_argument('-nr', '--nr_resnet', type=float, default=cfg['nr_resnet'], help="")
parser.add_argument('-nf', '--nr_filters', type=float, default=cfg['nr_filters'], help="")
parser.add_argument('-nlm', '--nr_logistic_mix', type=float, default=cfg['nr_logistic_mix'], help="")
parser.add_argument('-sr', '--sample_range', type=float, default=cfg['sample_range'], help="")
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
# new features
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Under debug mode?')
parser.add_argument('-os', '--one_stage', dest='one_stage', action='store_true', help='One stage training?')
parser.add_argument('-um', '--mode', type=str, default=cfg['mode'], help='')
parser.add_argument('-ns', '--network_type', type=str, default=cfg['network_type'], help='')

parser.add_argument('-pd', '--pvae_dir', type=str, default=cfg['pvae_dir'], help='')
parser.add_argument('-en', '--exp_name', type=str, default=cfg['exp_name'], help='')

args = parser.parse_args(sys.argv[2:])
if args.mode in ['eval', 'test', 'generate', 'inpainting', 'traverse', 'inspect', 'temp']:
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
    "nr_resnet": args.nr_resnet,
    "nr_filters": args.nr_filters,
    "nr_logistic_mix": args.nr_logistic_mix,
    "sample_range": args.sample_range,
    "network_type": args.network_type,
}

learner = ControlledPixelVAELearner(args.nr_gpu, args.save_dir, args.img_size, args.z_dim, args.phase, exp_name=args.exp_name)
learner.load_data(args.data_set, args.batch_size*args.nr_gpu, use_debug_mode=args.debug)
if args.phase == 'pvae':
    trainable_params = ["forward_pixel_cnn", "conv_encoder", "conv_decoder"]
elif args.phase == 'ce':
    trainable_params = ["forward_pixel_cnn", "reverse_pixel_cnn"]

eval_keys = ['total loss', 'nll loss', 'reg loss', 'bits per dim', 'mi']
learner.construct_models(model_cls=ControlledConvPixelVAE, model_opt=model_opt, learning_rate=args.learning_rate, trainable_params=trainable_params, eval_keys=eval_keys)

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
        if args.phase == 'ce':
            if not args.one_stage:
                learner.preload(from_dir=args.pvae_dir, var_list=get_trainable_variables(["forward_pixel_cnn", "conv_encoder", "conv_decoder"]))
        learner.train(**kwargs)
    elif args.mode == 'test':
        learner.eval(which_set='test', mgen=get_generator('bottom half', size=args.img_size), generate_samples=True)
    elif args.mode == 'inpainting':
        layout = (10, 10)
        same_inputs = False
        use_mask_at = "{0}_{1}.npz".format(args.mask_type, args.data_set)
        learner.inpainting(get_generator(args.mask_type, size=args.img_size), layout=layout, same_inputs=same_inputs, use_mask_at=use_mask_at)
    elif args.mode == 'traverse':
        mids = [5,6,7,8,9,10] #[7, 8, 11, 15]
        # mask_descriptions = ['mouth', 'eye', 'nose']
        mask_descriptions = ['mnist top 20']
        for mid in mids:
            for mask in mask_descriptions:
                print("mid {0}, mask {1}".format(mid, mask))
                learner.traverse(get_generator(mask, size=args.img_size), image_id=mid)
    elif args.mode == 'inspect':
        mids = [0, 1, 2, 3, 4, 5] #[7, 8, 11, 15]
        for mid in mids:
            print("mid {0}".format(mid))
            learner.inspect(image_id=mid, num_traversal_step=13)

    elif args.mode == 'temp':
        # learner.examine_two_stage(get_generator('random rec', size=args.img_size))
        learner.examine_reg(get_generator('transparent', size=args.img_size), image_id=7)
