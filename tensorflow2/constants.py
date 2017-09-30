import tensorflow as tf
EMOTIONS = ["neutral", "disgust", "happy"]


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return filename and example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 1])  # old
    # img = tf.reshape(img, [128*128])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # normalize
    label = tf.cast(features['label'], tf.int32)
    label = tf.sparse_to_dense(label, [3], 1, 0)
    return img, label


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_and_decode(filename_queue)
    min_after_dequeue = 1000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity, num_threads=num_threads,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch