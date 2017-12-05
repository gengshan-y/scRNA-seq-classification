from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import pdb
import log_init
from model import build_model, create_dataloader
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# Network Parameters
learning_rate = 1e-4

# Training Parameters
num_classes = 46
BATCH_SIZE=100
INPUT_PATH = os.path.join('..', 'data')

n_samples_t = 16747
n_samples_v = 4642

steps_epoch = int(n_samples_t / BATCH_SIZE / 100) * 100
num_steps = steps_epoch * 30
eval_iter = steps_epoch
VAL_ITER = int(n_samples_v / BATCH_SIZE)+1
save_step = steps_epoch

print('#steps/epoch=%d, #batches to validate=%d, #steps in total=%d'\
                    %(steps_epoch, VAL_ITER, num_steps))

import argparse
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--log_directory',           type=str,   help='log files and models to store', default='../log')
parser.add_argument('--data_name',           type=str,   help='path to a specific checkpoint to load', default='_full_norm')
args = parser.parse_args()

train_name = 'train' + args.data_name + '.tfrecords' 
val_name = 'val' + args.data_name + '.tfrecords'
sem_map_label = np.load('../data/sem_map.npy')[()]
sem_map_label_v = np.load('../data/sem_map_v.npy')[()]
map_label = np.load('../data/map.npy')[()]['n2l']
logger = log_init.make_logger('%s/log_'%args.log_directory)


data_batch_train, label_batch_train, weight_batch = create_dataloader(os.path.join(INPUT_PATH,\
                                               train_name), BATCH_SIZE)
data_batch_val, label_batch_val = create_dataloader(os.path.join(INPUT_PATH,\
                                       val_name), BATCH_SIZE, is_training=False)

# Construct model
y_pred,loss,y_true_t =  build_model(data_batch_train,label_batch_train, weights=weight_batch)
y_pred_v,loss_v,y_true_v =  build_model(data_batch_val,label_batch_val,is_train=False,reuse=True)

conf_v = tf.confusion_matrix(y_true_v, y_pred_v, num_classes)

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Start Training
# Start a new TF session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list='0'

#summary_op = tf.summary.merge_all()
summary_op = tf.summary.merge_all('train')
summary_op_v = tf.summary.merge_all('val')
train_saver = tf.train.Saver(max_to_keep=30)

with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter(args.log_directory, sess.graph)
    # Run the initializer
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    if args.checkpoint_path != '':
        train_saver.restore(sess, args.checkpoint_path)

    # Training
    for i in range(0, num_steps):
        # Prepare Data
        _, l, ypt,lbt = sess.run([optimizer, loss, y_pred,y_true_t])  # dont use eval()
        acc_t =  np.sum(ypt == lbt) / float(BATCH_SIZE)
        # Display logs per step
        l_vs = []; conf_v_all=np.zeros((num_classes, num_classes))
        if i % eval_iter == 0:
            for j in range(VAL_ITER):
                l_v,conf_v_batch = sess.run([loss_v, conf_v])
                conf_v_all += conf_v_batch
                l_vs.append(l_v)
            l_val = np.mean(l_vs)

            tp = np.diag(conf_v_all)
            precision = np.nan_to_num(tp / np.sum(conf_v_all, axis=0))
            recall = np.nan_to_num(tp/ np.sum(conf_v_all, axis=1))
            acc_v = sum(tp) / (BATCH_SIZE * VAL_ITER)
        
            logger.info(conf_v_all)
            for k in np.argsort(recall):
                v = map_label[k]
                logger.info ('class: %d/%s, prec/recall: %f/%f (%d/%d)' %\
                      (k, v, precision[k],recall[k], sem_map_label[k], sem_map_label_v[k]) )
            
            logger.debug('Step %i: Minibatch loss/acc: %f/%f, VAL loss/acc: %f/%f '\
                              % (i, l,acc_t, l_val, acc_v))
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, global_step=i)
            summary_str = sess.run(summary_op_v)
            summary_writer.add_summary(summary_str, global_step=i)
        if i and i % save_step == 0:
            train_saver.save(sess, args.log_directory + '/model', global_step=i)
    train_saver.save(sess, args.log_directory + '/model', global_step=i)
