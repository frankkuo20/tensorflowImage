import os

import tensorflow as tf
import math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = 128  # 图像大小
LABEL_CNT = 3  # 标签类别的数量

max_step = 200


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


class CnnObj:
    def __init__(self):
        x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])  # 128*128
        y_ = tf.placeholder(tf.float32, [None, LABEL_CNT])  # right answer

        # one
        W_conv = weight_variable([5, 5, 1, 32])

        b_conv = bias_variable([32])
        h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
        h_pool = max_pool_2x2(h_conv)

        # two
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # final 128/2/2 = 32
        W_fc = weight_variable([32 * 32 * 64, 1024])
        b_fc = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 64])
        h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

        W_fc2 = weight_variable([1024, LABEL_CNT])
        b_fc2 = bias_variable([LABEL_CNT])

        y = tf.matmul(h_fc_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()

        self.x = x
        self.y = y
        self.y_ = y_
        self.init = init
        self.keep_prob = keep_prob
        self.train_step = train_step
        self.accuracy = accuracy

        self.h_conv = h_conv
        self.h_conv2 = h_conv2
        self.W_conv = W_conv


    def getPredList(self, image):
        im_raw = image.tobytes()
        img = tf.decode_raw(im_raw, tf.uint8)
        img = tf.reshape(img, [IMG_SIZE, IMG_SIZE, 1])
        image = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # normalize

        x = self.x
        y = self.y
        keep_prob = self.keep_prob

        sess = tf.Session()
        saver = tf.train.Saver()
        tf.reset_default_graph()

        saver.restore(sess, './cnn_train/graph.ckpt-{}'.format(max_step))
        image = sess.run([image])
        result = sess.run(y, feed_dict={x: image, keep_prob: 1.0})
        return result

    def getPredNum(self, image):
        result = self.getPredList(image)
        resultNum = result.argmax()
        return resultNum


    # def plotNNFilter(self, units):
    #     filters = units.shape[3]
    #     plt.figure(1, figsize=(20, 20))
    #     n_columns = 6
    #     n_rows = math.ceil(filters / n_columns) + 1
    #     for i in range(filters):
    #         plt.subplot(n_rows, n_columns, i + 1)
    #         plt.title('Filter ' + str(i))
    #         plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
    #     plt.show()
    #
    # def getActivations(self, layer, stimuli):
    #     x = self.x
    #     units = sess.run(layer, feed_dict={x: stimuli, keep_prob: 1.0})
    #     plotNNFilter(units)
    #
    #     imageToUse = test_imgs
    #
    # plt.imshow(np.reshape(imageToUse, [128, 128]), interpolation="nearest", cmap="gray")
    # plt.show()
    # getActivations(h_conv, imageToUse)
    # getActivations(h_conv2, imageToUse)
    # getActivations(W_conv, imageToUse)
