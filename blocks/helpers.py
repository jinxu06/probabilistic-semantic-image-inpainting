import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from PIL import Image

def int_shape(x):
    return list(map(int, x.get_shape()))

def log_sum_exp(x, axis=-1):
    return tf.reduce_logsumexp(x, axis=axis)

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def PIL_to_uint(pil_img):
    pass

def uint_to_PIL(uint_img):
    pass

def PIL_to_float(pil_img):
    pass

def uint_to_float(pil_img):
    pass

def tile_images(imgs, size=(6, 6)):
    imgs = imgs[:size[0]*size[1], :, :, :]
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    all_images = np.zeros((img_h*size[0], img_w*size[1], 3), np.uint8)
    for j in range(size[0]):
        for i in range(size[1]):
            all_images[img_h*j:img_h*(j+1), img_w*i:img_w*(i+1), :] = imgs[j*size[1]+i, :, :, :]
    return all_images

def visualize_samples(images, name="results/test.png", layout=[5,5], vrange=[-1., 1.]):
    if images.shape[-1] == 1:
        images = np.concatenate([images for i in range(3)], axis=-1)
    images = (images - vrange[0]) / (vrange[1]-vrange[0]) * 255.
    images = np.rint(images).astype(np.uint8)
    view = tile_images(images, size=layout)
    if name is None:
        return view
    view = Image.fromarray(view, 'RGB')
    view.save(name)

def broadcast_masks_tf(masks, num_channels=None, batch_size=None):
    if num_channels is not None:
        masks = tf.stack([masks for i in range(num_channels)], axis=-1)
    if batch_size is not None:
        masks = tf.stack([masks for i in range(batch_size)], axis=0)
    return masks

def broadcast_masks_np(masks, num_channels=None, batch_size=None):
    if num_channels is not None:
        masks = np.stack([masks for i in range(num_channels)], axis=-1)
    if batch_size is not None:
        masks = np.stack([masks for i in range(batch_size)], axis=0)
    return masks


def get_trainable_variables(flist, filter_type="in"):
    all_vs = tf.trainable_variables()
    if filter_type=="in":
        vs = []
        for s in flist:
            vs += [p for p in all_vs if s in p.name]
    elif filter_type=="not in":
        vs = all_vs
        for s in flist:
            vs = [p for p in vs if s not in p.name]
    return vs

def get_nonlinearity(name):
    if name=="relu":
        return tf.nn.relu
    elif name=="elu":
        return tf.nn.elu
    elif name=='tanh':
        return tf.nn.tanh
    elif name=='sigmoid':
        return tf.sigmoid

def params_to_logits(x, params):
    bit_depth = 256
    r = tf.stack([x[:,:,:,0] for i in range(bit_depth)], axis=-1)
    g = tf.stack([x[:,:,:,1] for i in range(bit_depth)], axis=-1)
    b = tf.stack([x[:,:,:,2] for i in range(bit_depth)], axis=-1)
    logits_r = params[:,:,:,:bit_depth]
    logits_g = params[:,:,:,bit_depth:bit_depth*2]
    logits_b = params[:,:,:,bit_depth*2:bit_depth*3]
    alpha = params[:,:,:,bit_depth*3:bit_depth*4]
    beta = params[:,:,:,bit_depth*4:bit_depth*5]
    gamma = params[:,:,:,bit_depth*5:bit_depth*6]
    logits_g = logits_g + alpha * r
    logits_b = logits_b + beta * r + gamma * g
    logits = tf.stack([logits_r, logits_g, logits_b], axis=-2)
    return logits



# def mix_logistic_to_logits(l):
#     ls = int_shape(l)
#     all_log_probs = []
#     for i in range(256):
#         x = tf.ones(shape=ls[:3]+[3]) * i
#         log_probs = __mix_logistic_to_one_logit(x, l)
#         all_log_probs.append(log_probs)
#     return tf.stack(all_log_probs, axis=-1)
#
#
# def __mix_logistic_to_one_logit(x, l):
#     ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
#     xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
#     nr_mix = int(ls[-1] / 12) ## # here and below: unpacking the params of the mixture of logistics
#     logit_probs = tf.reshape(l[:,:,:,:3 * nr_mix], ls[:3]+[3,nr_mix]) # let's not share mix indicator
#     l = tf.reshape(l[:,:,:,3 * nr_mix:], xs + [nr_mix*3]) ##
#     means = l[:,:,:,:,:nr_mix]
#     log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
#     coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
#
#     x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
#     m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
#     m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
#     means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)
#     centered_x = x - means
#     inv_stdv = tf.exp(-log_scales)
#
#     plus_in = inv_stdv * (centered_x + 1./255.)
#     cdf_plus = tf.nn.sigmoid(plus_in)
#     min_in = inv_stdv * (centered_x - 1./255.)
#     cdf_min = tf.nn.sigmoid(min_in)
#     log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
#     log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
#     cdf_delta = cdf_plus - cdf_min # probability for all other cases
#     mid_in = inv_stdv * centered_x
#     log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)
#
#     log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
#     log_probs = log_probs + log_prob_from_logits(logit_probs)
#     # log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
#     lse = log_sum_exp(log_probs)
#     return lse

class Monitor(object):

    def __init__(self, dict={}, config_str="config not given", log_file_path="logfile"):
        self.dict = dict
        self.keys = self.dict.keys()
        self.fetches = self.__fetches(self.keys)
        self.cur_values = []
        self.epoch_values = []
        self.past_epoch_stats = []
        self.num_epoches = 0
        self.log_file_path = log_file_path
        with open(self.log_file_path, "w") as f:
            f.write(config_str+"\n")

    def __fetches(self, keys):
        fetches = []
        for key in keys:
            fetches.append(self.dict[key])
        return fetches

    def evaluate(self, sess, feed_dict):
        self.cur_values = sess.run(self.fetches, feed_dict=feed_dict)
        self.epoch_values.append(self.cur_values)

    def summarise_epoch(self, keys=None, time=0., log=True):
        epoch_values = np.array(self.epoch_values)
        stats = np.mean(epoch_values, axis=0)
        self.past_epoch_stats.append(stats)
        s = self.__display(stats, keys, time)
        print(s)
        sys.stdout.flush()
        if log:
            with open(self.log_file_path, "a") as f:
                f.write(s+"\n")
        self.epoch_values = []
        self.num_epoches += 1

    def __display(self, stats, keys=None, time=0.):
        if keys is None:
            keys = self.keys
        results = {}
        for k, s in zip(self.keys, stats):
            results[k] = s
        ret_str = "* epoch {0} {1} -- ".format(self.num_epoches, "{"+"%0.2f"%time+"s}")
        for key in keys:
            ret_str += "{0}:{1:.3f}   ".format(key, results[key])
        return ret_str

# class Recorder(object):
#
#     def __init__(self, dict={}, config_str="config not given", log_file="temp"):
#         self.dict = dict
#         self.keys = self.dict.keys()
#         self.fetches = self.__fetches(self.keys)
#         self.cur_values = []
#         self.epoch_values = []
#         self.past_epoch_stats = []
#         self.num_epoches = 0
#         self.log_file = log_file
#         with open(self.log_file, "w") as f:
#             f.write(config_str+"\n")
#
#     def __fetches(self, keys):
#         fetches = []
#         for key in keys:
#             fetches.append(self.dict[key])
#         return fetches
#
#     def evaluate(self, sess, feed_dict):
#         self.cur_values = sess.run(self.fetches, feed_dict=feed_dict)
#         self.epoch_values.append(self.cur_values)
#
#     def finish_epoch_and_display(self, keys=None, time=0., log=True):
#         epoch_values = np.array(self.epoch_values)
#         stats = np.mean(epoch_values, axis=0)
#         self.past_epoch_stats.append(stats)
#         s = self.__display(stats, keys, time)
#         print(s)
#         sys.stdout.flush()
#         with open(self.log_file, "a") as f:
#             f.write(s+"\n")
#         self.epoch_values = []
#         self.num_epoches += 1
#
#     def __display(self, stats, keys=None, time=0.):
#         if keys is None:
#             keys = self.keys
#         results = {}
#         for k, s in zip(self.keys, stats):
#             results[k] = s
#         ret_str = "* epoch {0} {1} -- ".format(self.num_epoches, "{"+"%0.2f"%time+"s}")
#         for key in keys:
#             ret_str += "{0}:{1:.3f}   ".format(key, results[key])
#         return ret_str
