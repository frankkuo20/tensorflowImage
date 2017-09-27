import os

import tensorflow as tf
from constants import EMOTIONS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    label = tf.sparse_to_dense(label, [4], 1, 0)
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


# 获取batch。注意这里是tensor，需要运行
img_batch, label_batch = input_pipeline(["./0_train/train.bin"], 50)

# =====================================================================================
IMG_SIZE = 128  # 图像大小
LABEL_CNT = 4  # 标签类别的数量


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# with tf.name_scope('Inputs'):
x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1], name='x')  # 128*128
y_ = tf.placeholder(tf.float32, [None, LABEL_CNT], name='y_')  # right answer

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
W_fc = weight_variable([32*32*64, 1024])
b_fc = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

keep_prob = tf.placeholder(tf.float32)
h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])

y = tf.matmul(h_fc_drop, W_fc2) + b_fc2

# with tf.name_scope('Loss'):
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# 梯度下降法gradient descent algorithm
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# with tf.name_scope('Trains'):
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

disp_step = 5
save_step = 100
max_step = 200  # 最大迭代次数
step = 0
saver = tf.train.Saver()  # 用来保存模型的
epoch = 5


with tf.Session() as sess:
    # tensorboard
    writer = tf.summary.FileWriter("logs/", graph=sess.graph)

    coord = tf.train.Coordinator()
    sess.run(init)

    # start_queue_runnes读取数据，具体用法参见官网
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        # 获取训练数据成功，并且没有到达最大训练次数
        while not coord.should_stop() and step < max_step:
            step += 1
            # 运行tensor，获取数据
            imgs, labels = sess.run([img_batch, label_batch])
            # 训练。训练时dropout层要有值。
            sess.run(train_step, feed_dict={x: imgs, y_: labels, keep_prob: 1.0})
            if step % epoch == 0:  # step
                # 输出当前batch的精度。预测时keep的取值均为1
                acc = sess.run(accuracy, feed_dict={x: imgs, y_: labels, keep_prob: 1.0})
                print('%s accuracy is %.2f' % (step, acc))
            if step % save_step == 0:
                # 保存当前模型
                save_path = saver.save(sess, './cnn_train/graph.ckpt', global_step=step)
                print("save graph to %s" % save_path)
    except tf.errors.OutOfRangeError as e:
        print("reach epoch limit")
        print(e)
    except Exception as e:
        print('eee')
        print(e)
    finally:
        coord.request_stop()
    coord.join(threads)
    save_path = saver.save(sess, './cnn_train/graph.ckpt', global_step=step)

print("training is done")


test_img_batch, test_label_batch = input_pipeline(["./0_train/test.bin"], 5)
with tf.Session() as sess:
    # 加载模型。模型的文件名称看下本地情况
    print('./cnn_train/graph.ckpt-{}'.format(step))
    saver.restore(sess, './cnn_train/graph.ckpt-{}'.format(step))

    coord_test = tf.train.Coordinator()
    threads_test = tf.train.start_queue_runners(coord=coord_test)
    test_imgs, test_labels = sess.run([test_img_batch, test_label_batch])
    # 预测阶段，keep取值均为1
    acc = sess.run(accuracy, feed_dict={x: test_imgs, y_: test_labels, keep_prob: 1.0})
    print("predict accuracy is %.2f" % acc)
    coord_test.request_stop()
    coord_test.join(threads_test)
