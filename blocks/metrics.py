import tensorflow as tf
import numpy as np

# def nats_per_dim(nll, dim):
#     -nll / dim + np.log(128)
#
# def nats_to_bits(nats):
#     return nats / np.log(2)
#
# def bits_per_dim(nll, dim):
#     return nats_to_bits(nats_per_dim(nll, dim))
#
# def nats_per_dim_tf(nll, dim):
#     return -nll / dim + tf.math.log(128.)
#
# def nats_to_bits_tf(nats):
#     return nats / tf.math.log(2.)
#
# def bits_per_dim_tf(nll, dim):
#     return nats_to_bits_tf(nats_per_dim_tf(nll, dim))

def bits_per_dim_tf(nll, dim, unit=1.0):
    # https://www.reddit.com/r/MachineLearning/comments/56m5o2/discussion_calculation_of_bitsdims/
    return (nll / dim + tf.math.log(unit)) / tf.math.log(2.)
