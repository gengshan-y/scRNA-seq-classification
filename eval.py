import pdb
import numpy as np
import os
import argparse
import tensorflow as tf
from model import build_model_test, create_dataloader_test

INPUT_PATH = os.path.join('..', 'data')
num_classes = 46
val_name = 'val_full_norm' + '.tfrecords'
num_samples = 4642
#val_name = 'test_full_norm' + '.tfrecords'
#num_samples = 2855


parser = argparse.ArgumentParser(description='.')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
args = parser.parse_args()

data_batch_val = create_dataloader_test(os.path.join(INPUT_PATH,val_name))
y_pred_v =  build_model_test(data_batch_val)

train_saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list='1'

ls = []
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    train_saver.restore(sess, args.checkpoint_path)
    
    pdb.set_trace()
    for i in range(num_samples):
        print i
        #print np.sum(sess.run([data_batch_val]))
        y_pred_batch_v = sess.run([y_pred_v])
        ls.append(y_pred_batch_v[0][0])

np.save('../pred_nnsoft_val.npy',ls)
