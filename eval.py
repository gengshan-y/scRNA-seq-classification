import pdb
from main import build_model
import argparse
import tensorflow as tf

BATCH_SIZE=100
INPUT_PATH = os.path.join('..', 'data')
val_name = 'val' + '.tfrecords'


parser = argparse.ArgumentParser(description='.')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
args = parser.parse_args()
pdb.set_trace()

data_batch_val, label_batch_val = create_dataloader(os.path.join(INPUT_PATH,\
                                    val_name), BATCH_SIZE, is_training=False)
y_pred_v,loss_v,y_true_v =  build_model(data_batch_val,label_batch_val,is_train=False)

train_saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list='0'

with tf.Session(config=config) as sess:
    train_saver.restore(sess, args.checkpoint_path)
