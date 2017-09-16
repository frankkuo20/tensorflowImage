import glob

import cv2
import tensorflow as tf
from constants import EMOTIONS
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def getPredNum(image):
    im_raw = image.tobytes()
    img = tf.decode_raw(im_raw, tf.uint8)
    img = tf.reshape(img, [128 * 128])
    image = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # normalize

    IMG_SIZE = 128
    LABEL_CNT = 4
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE])
    y_ = tf.placeholder(tf.float32, [None, LABEL_CNT])  # right answer

    W = tf.Variable(tf.zeros([IMG_SIZE * IMG_SIZE, LABEL_CNT]))
    b = tf.Variable(tf.zeros([LABEL_CNT]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)


    sess = tf.Session()
    saver = tf.train.Saver()
    tf.reset_default_graph()

    saver.restore(sess, './0_train/graph.ckpt-{}'.format(1000))
    image = sess.run(image)
    result = sess.run(y, feed_dict={x: [image]})
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












