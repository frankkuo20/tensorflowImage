import glob

import cv2
import tensorflow as tf
from constants import EMOTIONS
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def getPredNum(image):
    im_raw = image.tobytes()
    img = tf.decode_raw(im_raw, tf.uint8)
    img = tf.reshape(img, [128, 128, 1])
    image = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # normalize

    IMG_SIZE = 128
    LABEL_CNT = 4
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

    # final
    W_fc = weight_variable([32 * 32 * 64, 1024])
    b_fc = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 64])
    h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

    keep_prob = tf.placeholder(tf.float32)
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

    W_fc2 = weight_variable([1024, 4])
    b_fc2 = bias_variable([4])

    y = tf.matmul(h_fc_drop, W_fc2) + b_fc2


    sess = tf.Session()
    saver = tf.train.Saver()
    tf.reset_default_graph()

    saver.restore(sess, './final_train/graph.ckpt-{}'.format(1000))
    image = sess.run(image)
    result = sess.run(y, feed_dict={x: [image], keep_prob: 1.0})
    resultNum = result.argmax()
    return resultNum

IMAGE_PATH = '../dataset2/*'

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

imagePaths = glob.glob(IMAGE_PATH)

for imagePath in imagePaths:

    image = cv2.imread(imagePath)
    frame = image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        grayCut = gray[y:y + h, x:x + w]
        image = cv2.resize(grayCut, (128, 128))

        predNum = getPredNum(image)
        text = EMOTIONS[predNum]
        print(imagePath)
        print(text)












