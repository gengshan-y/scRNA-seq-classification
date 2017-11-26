import pdb
import tensorflow as tf

#fqueue = tf.train.string_input_producer(["../data/test_covariates.tsv"])
# reader = tf.TextLineReader(skip_header_lines=1)
#_, value = reader.read(fqueue)
#content = tf.decode_csv(value,[[0.]]*20499,field_delim='\t')
#features = tf.stack(content)

filename_queue = tf.train.string_input_producer(['../data/output_file_small.tfrecords'])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
        'label': tf.FixedLenFeature([], tf.int64),
        })

# data = tf.decode_raw(features['data'], tf.float32)
data = tf.cast(features['label'], tf.int32)


#data_batch = tf.train.shuffle_batch([features], batch_size=1, capacity=20, min_after_dequeue=10)
data_batch = tf.train.shuffle_batch([data], batch_size=1, capacity=20, min_after_dequeue=10)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    pdb.set_trace()
    x = sess.run([data_batch])
    
    coord.request_stop()
    coord.join(threads)
