from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import numpy as np
import pdb

# Training Parameters
learning_rate = 1e-4
num_steps = 30000

display_step = 100
save_step = 10000

# Network Parameters
num_input = 20463 

BATCH_SIZE=100
log_directory = '..'
INPUT_PATH = os.path.join('..', 'data')
VAL_ITER = 40  # cover all validation set

import argparse
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--data_name',           type=str,   help='path to a specific checkpoint to load', default='')
args = parser.parse_args()

train_name = 'train' + args.data_name + '.tfrecords' 
val_name = 'val' + args.data_name + '.tfrecords'

def create_dataloader(datapath, batch_size):
    filename_queue = tf.train.string_input_producer([datapath])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
        'data': tf.FixedLenFeature([num_input], tf.float32),
        'label': tf.FixedLenFeature([], tf.int64),
        })
    data = features['data']
    label = features['label']
    data_batch,label_batch = tf.train.shuffle_batch([data,label], batch_size=batch_size,\
                  capacity=batch_size*20, min_after_dequeue=batch_size*10)
    return data_batch, label_batch
    

data_batch_train, label_batch_train = create_dataloader(os.path.join(INPUT_PATH,\
                                               train_name), BATCH_SIZE)
data_batch_val, label_batch_val = create_dataloader(os.path.join(INPUT_PATH,\
                                               val_name), BATCH_SIZE)
X = tf.placeholder(tf.float32, shape=(BATCH_SIZE, num_input))
y_true = tf.placeholder(tf.int64, shape=(BATCH_SIZE))

# Construct model
def build_model(X,y_true,is_train=True,reuse=None,scope='model'):
    reg = 0.
    dropout = 0.7
    #with tf.variable_scope(scope, 'my_model' ,[X,y_true], reuse=reuse):
    with tf.variable_scope(scope, reuse=reuse):
        fc1 = tf.contrib.layers.fully_connected(X, 1024, weights_regularizer = layers.l2_regularizer(scale=reg))
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_train)
        fc2 = tf.contrib.layers.fully_connected(fc1, 100, weights_regularizer = layers.l2_regularizer(scale=reg))
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_train)
        #fc3 = tf.contrib.layers.fully_connected(fc2, 46)
        #fc1 = tf.contrib.layers.fully_connected(X, 1024, activation_fn=tf.nn.elu,normalizer_fn = slim.batch_norm)
        #fc2 = tf.contrib.layers.fully_connected(X, 100, activation_fn=tf.nn.elu,normalizer_fn = slim.batch_norm)
        fc3 = tf.contrib.layers.fully_connected(fc2, 46, activation_fn=None, weights_regularizer = layers.l2_regularizer(scale=reg))
        fc3 = tf.layers.dropout(fc3, rate=dropout, training=is_train)
        logits = tf.nn.softmax(fc3)
        y_pred = tf.argmax(logits,axis=-1)
        sup_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3, labels=y_true)
        sup_loss = tf.reduce_mean(sup_loss)

        # regularization
        reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'model')
        reg_ws = tf.reduce_sum(reg_ws)
        loss = sup_loss + reg_ws
        # loss = -tf.reduce_mean(tf.one_hot(y_true,46) * tf.log(logits))

        # build summaries
        if is_train:
            sum_name = 'train'
        else:
            sum_name = 'val'
        tf.summary.histogram('X', X, [sum_name])
        tf.summary.histogram('fc1', fc1, [sum_name])
        tf.summary.histogram('fc2', fc2, [sum_name])
        tf.summary.histogram('fc3', fc3, [sum_name])
        tf.summary.histogram('y_pred', y_pred, [sum_name])
        tf.summary.scalar('loss', loss, [sum_name])
        tf.summary.scalar('sup_loss', sup_loss, [sum_name])
        tf.summary.scalar('reg_loss', reg_ws, [sum_name])
    return y_pred, loss, y_true

y_pred,loss,_ =  build_model(X,y_true)
#y_pred,loss,_ =  build_model(data_batch_train,label_batch_train)
#y_pred_v,loss_v,y_true_v =  build_model(data_batch_val,label_batch_val,is_train=False,reuse=True)

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Start Training
# Start a new TF session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list='3'

#summary_op = tf.summary.merge_all()
summary_op = tf.summary.merge_all('train')
# summary_op_v = tf.summary.merge_all('val')
# train_saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter(log_directory + '/' + 'log', sess.graph)
    # Run the initializer
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    if args.checkpoint_path != '':
        train_saver.restore(sess, args.checkpoint_path.split(".")[0])

    # Training
    for i in range(0, num_steps):
        # Prepare Data
        dbt, lbt = sess.run([data_batch_train, label_batch_train])
        _, l, ypt = sess.run([optimizer, loss, y_pred],feed_dict={X:dbt,y_true:lbt})  # dont use eval()
        acc_t =  np.sum(ypt == lbt) / float(BATCH_SIZE)
        # Display logs per step
        l_vs = []; corr_num = 0
        if i % display_step == 0:
            for j in range(VAL_ITER):
                dbv, lbv = sess.run([data_batch_val, label_batch_val])
                l_v,ypv = sess.run([loss, y_pred],feed_dict={X:dbv,y_true:lbv})
                corr_num += np.sum(ypv == lbv)
                l_vs.append(l_v)
            l_val = np.mean(l_vs)
            acc_v = float(corr_num) / (BATCH_SIZE * VAL_ITER)
            print('Step %i: Minibatch loss/acc: %f/%f, VAL loss/acc: %f/%f '\
                              % (i, l,acc_t, l_val, acc_v))
            summary_str = sess.run(summary_op, feed_dict={X:dbt,y_true:lbt})
            summary_writer.add_summary(summary_str, global_step=i)
            #summary_str = sess.run(summary_op_v)
            #summary_writer.add_summary(summary_str, global_step=i)
        if i and i % save_step == 0:
            train_saver.save(sess, log_directory + '/model', global_step=i)
    train_saver.save(sess, log_directory + '/model', global_step=i)
