import glob
import os
import cv2
import tensorflow as tf

from tensorflow2.cnn import CnnObj
from tensorflow2.constants import EMOTIONS, IMG_SIZE, LABEL_CNT

max_step = 200


def getPredNum(image):
    im_raw = image.tobytes()
    img = tf.decode_raw(im_raw, tf.uint8)
    img = tf.reshape(img, [128, 128, 1])
    image = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # normalize

    cnnObj = CnnObj()
    x = cnnObj.x
    y = cnnObj.y
    keep_prob = cnnObj.keep_prob

    sess = tf.Session()
    saver = tf.train.Saver()

    tf.reset_default_graph()
    saver.restore(sess, './cnn_train/graph.ckpt-{}'.format(max_step))
    image = sess.run([image])
    result = sess.run(y, feed_dict={x: image, keep_prob: 1.0})
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

    for (x2, y2, w2, h2) in faces:
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
        grayCut = gray[y2:y2 + h2, x2:x2 + w2]
        image = cv2.resize(grayCut, (128, 128))

        predNum = getPredNum(image)
        text = EMOTIONS[predNum]
        print(imagePath)
        print(text)












