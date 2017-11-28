from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import pdb

# Training Parameters
learning_rate = 1e-3
num_steps = 30000

display_step = 100
save_step = 10000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 1 # 1st layer num features
num_hidden_2 = 1 # 2nd layer num features (the latent dim)
num_input = 20499 # MNIST data input (img shape: 28*28)



weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}


def create_dataloader(datapath, batch_size):
    filename_queue = tf.train.string_input_producer([datapath])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
        'data': tf.FixedLenFeature([20499], tf.float32),
        'label': tf.FixedLenFeature([], tf.int64),
        })
    data = features['data']
    label = features['label']
    data_batch,label_batch = tf.train.shuffle_batch([data,label], batch_size=batch_size,\
                  capacity=batch_size*20, min_after_dequeue=batch_size*10)
    return data_batch, label_batch
    

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


BATCH_SIZE=10
log_directory = '..'
INPUT_PATH = os.path.join('..', 'data')
data_batch, label_batch = create_dataloader(os.path.join(INPUT_PATH, 'output_file_small.tfrecords'), BATCH_SIZE)

X = data_batch

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op

# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


tf.summary.histogram('y_pred', y_pred, ['model'])
tf.summary.histogram('X', X, ['model'])
tf.summary.scalar('loss', loss, ['model'])



# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list='3'

pdb.set_trace()
summary_op = tf.summary.merge_all('model')
train_saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter(log_directory + '/' + 'log', sess.graph)

    # Run the initializer
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Training
    for i in range(0, num_steps):

        # Prepare Data
        _, l = sess.run([optimizer, loss])
        # Display logs per step
        if i and i % display_step == 0:
            print('Step %i: Minibatch Loss: %f' % (i, l))
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, global_step=i)
        if i and i % save_step == 0:
            train_saver.save(sess, log_directory + '/model', global_step=i)
    train_saver.save(sess, log_directory + '/model', global_step=i)


#    # Testing
#    # Encode and decode images from test set and visualize their reconstruction.
#    n = 4
#    canvas_orig = np.empty((28 * n, 28 * n))
#    canvas_recon = np.empty((28 * n, 28 * n))
#    for i in range(n):
#        # MNIST test set
#        batch_x, _ = mnist.test.next_batch(n)
#        # Encode and decode the digit image
#        g = sess.run(decoder_op, feed_dict={X: batch_x})
#
#        # Display original images
#        for j in range(n):
#            # Draw the original digits
#            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
#                batch_x[j].reshape([28, 28])
#        # Display reconstructed images
#        for j in range(n):
#            # Draw the reconstructed digits
#            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
#                g[j].reshape([28, 28])
#
#    print("Original Images")
#    plt.figure(figsize=(n, n))
#    plt.imshow(canvas_orig, origin="upper", cmap="gray")
#    plt.show()
#
#    print("Reconstructed Images")
#    plt.figure(figsize=(n, n))
#    plt.imshow(canvas_recon, origin="upper", cmap="gray")
#    plt.show()
