from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pdb

# Training Parameters
learning_rate = 1e-4
num_steps = 30000

display_step = 100
save_step = 10000

# Network Parameters
num_hidden_1 = 1 # 1st layer num features
num_hidden_2 = 1 # 2nd layer num features (the latent dim)
num_input = 20463 # MNIST data input (img shape: 28*28)

BATCH_SIZE=100
log_directory = '..'
INPUT_PATH = os.path.join('..', 'data')
VAL_ITER = 30

import argparse
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
args = parser.parse_args()


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
                                               'train.tfrecords'), BATCH_SIZE)
data_batch_val, label_batch_val = create_dataloader(os.path.join(INPUT_PATH,\
                                               'val.tfrecords'), BATCH_SIZE)
#X = tf.placeholder(tf.float32, shape=(BATCH_SIZE, num_input))
#y_true = tf.placeholder(tf.int64, shape=(BATCH_SIZE))
X = data_batch_train
y_true = label_batch_train
# Construct model
fc1 = tf.contrib.layers.fully_connected(X, 1024)
fc2 = tf.contrib.layers.fully_connected(fc1, 100)
fc3 = tf.contrib.layers.fully_connected(fc1, 46)
#fc1 = tf.contrib.layers.fully_connected(X, 1024, activation_fn=tf.nn.elu)
#fc2 = tf.contrib.layers.fully_connected(fc1, 100, activation_fn=tf.nn.elu)
#fc3 = tf.contrib.layers.fully_connected(fc1, 46, activation_fn=tf.nn.sigmoid, normalizer_fn = slim.batch_norm)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc3,\
                  labels=tf.one_hot(y_true,46))
loss = tf.reduce_mean(loss)
y_pred = tf.nn.softmax(fc3)
y_pred = tf.argmax(y_pred,axis=-1)

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

tf.summary.histogram('X', X, ['model'])
tf.summary.histogram('fc1', fc1, ['model'])
tf.summary.histogram('fc2', fc2, ['model'])
tf.summary.histogram('fc3', fc3, ['model'])
tf.summary.histogram('y_pred', y_pred, ['model'])
tf.summary.scalar('loss', loss, ['model'])

# Start Training
# Start a new TF session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list='3'

summary_op = tf.summary.merge_all('model')
train_saver = tf.train.Saver()

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
        _, l = sess.run([optimizer, loss])
        #_, l = sess.run([optimizer, loss],feed_dict={
        #              X:data_batch_train.eval(), 
        #         y_true:label_batch_train.eval()})
        # Display logs per step
        if i % display_step == 0:
            print('Step %i: Minibatch Loss: %f' % (i, l))
            #corr_num = 0
            #for j in range(VAL_ITER):
            #    numy_true,numy_pred = sess.run([y_true,y_pred],feed_dict={
            #                X:data_batch_val.eval(),
            #           y_true:label_batch_val.eval()})
            #    corr_num += np.sum(numy_true == numy_pred)
            #acc_show = float(corr_num)/(VAL_ITER*BATCH_SIZE)
            #print('Step %i: Minibatch Loss: %f, VAL acc: %f' % (i, l, acc_show))
            #summary_str = sess.run(summary_op, feed_dict={
            #          X:data_batch_train.eval(),
            #     y_true:label_batch_train.eval()})
            #summary_writer.add_summary(summary_str, global_step=i)
        if i and i % save_step == 0:
            train_saver.save(sess, log_directory + '/model', global_step=i)
    train_saver.save(sess, log_directory + '/model', global_step=i)

