import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

num_input = 20463
num_classes = 46

# Construct model
def build_model(X,y_true,is_train=True,reuse=None,scope='model', weights=None):
    reg = 0.
    dropout = 0.2
    with tf.variable_scope(scope, reuse=reuse):
        fc1 = tf.contrib.layers.fully_connected(X, 2048, activation_fn=tf.nn.elu, \
                             weights_regularizer = layers.l2_regularizer(scale=reg))
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_train)

        fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.elu, \
                             weights_regularizer = layers.l2_regularizer(scale=reg))
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_train)

        fc3 = tf.contrib.layers.fully_connected(fc2, 100, activation_fn=tf.nn.elu, \
                             weights_regularizer = layers.l2_regularizer(scale=reg))
        fc3 = tf.layers.dropout(fc3, rate=dropout, training=is_train)

        fc_l = tf.contrib.layers.fully_connected(fc3, num_classes, activation_fn=None, \
                             weights_regularizer = layers.l2_regularizer(scale=reg))
        fc_l = tf.layers.dropout(fc_l, rate=dropout, training=is_train)

        logits = tf.nn.softmax(fc_l)
        y_pred = tf.argmax(logits,axis=-1)
        sup_loss_all = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_l, labels=y_true)
        # sup_loss_all = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.one_hot(y_true, num_classes), logits=fc_l), 1)
        if not weights is None:
            sup_loss_all *= weights
        sup_loss = tf.reduce_mean(sup_loss_all)

        # regularization
        reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'model')
        reg_ws = tf.reduce_sum(reg_ws)
        loss = sup_loss + reg_ws

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
        tf.summary.histogram('sup_loss_all', sup_loss_all, [sum_name])
        tf.summary.scalar('loss', loss, [sum_name])
        tf.summary.scalar('sup_loss', sup_loss, [sum_name])
        tf.summary.scalar('reg_loss', reg_ws, [sum_name])
    return y_pred, loss, y_true



def create_dataloader(datapath, batch_size, is_training=True):
    print('reading from %s'%datapath)
    filename_queue = tf.train.string_input_producer([datapath])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if is_training:
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
            'data': tf.FixedLenFeature([num_input], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
            'weight': tf.FixedLenFeature([], tf.float32)
            })
        data = features['data']
        label = features['label']
        weight = features['weight']
        data_batch,label_batch,weight_batch = tf.train.shuffle_batch([data,label,weight], batch_size=batch_size,\
                  capacity=batch_size*20, min_after_dequeue=batch_size*10)
        return data_batch, label_batch, weight_batch
    else:
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
            'data': tf.FixedLenFeature([num_input], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
            })
        data = features['data']
        label = features['label']
        data_batch,label_batch = tf.train.batch([data,label], batch_size=batch_size,\
                  capacity=batch_size*20)
        return data_batch, label_batch


def create_dataloader_test(datapath):
    print('reading from %s'%datapath)
    #filename_queue = tf.train.string_input_producer([datapath], num_epochs=1, shuffle=False)
    filename_queue = tf.train.string_input_producer([datapath], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'data': tf.FixedLenFeature([num_input], tf.float32),
        })
    data = features['data']
    data_batch = tf.train.batch([data], batch_size=1,\
                  capacity=20)
    return data_batch
    #return tf.expand_dims(data, 0)

# Construct model
def build_model_test(X):
    is_train=False
    reg = 0.
    dropout = 0.2
    with tf.variable_scope('model', reuse=None):
        fc1 = tf.contrib.layers.fully_connected(X, 2048, activation_fn=tf.nn.elu, \
                             weights_regularizer = layers.l2_regularizer(scale=reg))
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_train)

        fc2 = tf.contrib.layers.fully_connected(fc1, 512, activation_fn=tf.nn.elu, \
                             weights_regularizer = layers.l2_regularizer(scale=reg))
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_train)

        fc3 = tf.contrib.layers.fully_connected(fc2, 100, activation_fn=tf.nn.elu, \
                             weights_regularizer = layers.l2_regularizer(scale=reg))
        fc3 = tf.layers.dropout(fc3, rate=dropout, training=is_train)

        fc_l = tf.contrib.layers.fully_connected(fc3, num_classes, activation_fn=None, \
                             weights_regularizer = layers.l2_regularizer(scale=reg))
        fc_l = tf.layers.dropout(fc_l, rate=dropout, training=is_train)

        logits = tf.nn.softmax(fc_l)
        y_pred = tf.argmax(logits,axis=-1)
    return y_pred
