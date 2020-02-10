from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

from supervised_models import SupervisedGraphsage
from models import SAGEInfo
from minibatch import NodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from utils import load_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
# flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('train_prefix', './example_data/ppi', 'prefix identifying training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')  #BR check this?
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
# flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_integer('identity_dim', 1, 'Set to positive value to use identity embedding features of that dimension. Default 0.') #BR :Coz of error in supervised_models.py, line 57.

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
# flags.DEFINE_string('base_log_dir', 'log', 'base directory for logging and saving embeddings') #BR
flags.DEFINE_integer('validate_iter', 100, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.compat.v1.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.compat.v1.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data):

    G = train_data[0]
    features = train_data[1]
    labels = train_data[2]
    train_nodes = train_data[3]
    test_nodes = train_data[4]
    val_nodes = train_data[5]
    num_classes = 2

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G,
            placeholders,
            labels,
            train_nodes,
            test_nodes,
            val_nodes,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                           SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

    model = SupervisedGraphsage(num_classes, placeholders,
                                features,
                                adj_info,
                                minibatch.deg,
                                layer_infos,
                                model_size=FLAGS.model_size,
                                sigmoid_loss=FLAGS.sigmoid,
                                identity_dim=FLAGS.identity_dim,
                                logging=True)

    print('x')


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")
    train(train_data)


if __name__ == '__main__':
    tf.app.run()