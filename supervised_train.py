from __future__ import division
from __future__ import print_function

import os
import csv
import time
import datetime
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
from sklearn import metrics

from supervised_models import SupervisedGraphsage
from models import SAGEInfo
from minibatch import NodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from utils import load_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
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
# flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('batch_size', 2, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
# flags.DEFINE_integer('identity_dim', 1, 'Set to positive value to use identity embedding features of that dimension. Default 0.') #BR :Coz of error in supervised_models.py, line 57.

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 100, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8


def eval(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

    f1_score = metrics.f1_score(y_true, y_pred)
    # conf_mat = metrics.confusion_matrix(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    return acc, prec, rec, f1_score
    # return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels, batch = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss], #BR: Prediction probability values
                        feed_dict=feed_dict_val)

    with open('labels_and_preds.txt', 'w') as f:
        writer = csv.writer(f)
        for i in range(len(batch)):
            l = []
            l.append(batch[i])
            l.extend(labels[i])
            l.extend(node_outs_val[0][i])
            writer.writerow(l)

    acc, prec, rec, f1_score = eval(labels, node_outs_val[0])
    return node_outs_val[1], acc, prec, rec, f1_score
    # mic, mac = calc_f1(labels, node_outs_val[0])
    # return node_outs_val[1], mic, mac, (time.time() - t_test)

def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}__{time:s}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate,
            time=str(datetime.datetime.now()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)

    acc, prec, rec, f1_score = eval(labels, val_preds)
    return np.mean(val_losses), acc, prec, rec, f1_score
    # f1_scores = calc_f1(labels, val_preds)
    # return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)



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


    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    log_folder = log_dir()
    summary_writer = tf.summary.FileWriter(log_folder, sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    # Train model

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels, batch = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict) #BR: See results
            train_cost = outs[2]

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    # val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch,
                    #                                                                   FLAGS.batch_size)
                    val_cost,  acc, prec, rec, f1_score = incremental_evaluate(sess, model, minibatch,
                                                                                      FLAGS.batch_size)
                else:
                    # val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch,
                    #                                                       FLAGS.validate_batch_size)
                    val_cost,  acc, prec, rec, f1_score = evaluate(sess, model, minibatch,
                                                                             FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                # train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                train_acc, train_prec, train_rec, train_f1_score = eval(labels, outs[-1])
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_accuracy=", "{:.5f}".format(train_acc),
                      "train_precision=", "{:.5f}".format(train_prec),
                      "train_recall=", "{:.5f}".format(train_rec),
                      "train_f1_score=", "{:.5f}".format(train_f1_score),
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_accuracy=", "{:.5f}".format(acc),
                      "val_precision=", "{:.5f}".format(prec),
                      "val_recall=", "{:.5f}".format(rec),
                      "val_f1_score=", "{:.5f}".format(f1_score))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break

    print("Optimization Finished!")
    sess.run(val_adj_info.op)
    # val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
    val_cost,  acc, prec, rec, f1_score = incremental_evaluate(sess, model, minibatch,
                                                                        FLAGS.batch_size)
    print("Full validation stats:",
          "loss=", "{:.5f}".format(val_cost),
          "accuracy=", "{:.5f}".format(acc),
          "precision=", "{:.5f}".format(prec),
          "recall=", "{:.5f}".format(rec),
          "f1_score=", "{:.5f}".format(f1_score))
    with open(log_folder + "val_stats.txt", "w") as fp:
        fp.write("loss={:.5f} accuracy={:.5f} precision={:.5f} recall={:.5f} f1_score={:.5f}".
                 format(val_cost, acc, prec, rec, f1_score))

    print("Writing test set stats to file")
    # val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size,
    #                                                                   test=True)
    val_cost,  acc, prec, rec, f1_score = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size,
                                                                      test=True)
    with open(log_folder + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} accuracy={:.5f} precision={:.5f} recall={:.5f} f1_score={:.5f}".
                 format(val_cost, acc, prec, rec, f1_score))


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")
    train(train_data)


if __name__ == '__main__':
    tf.app.run()