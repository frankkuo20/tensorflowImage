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

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    # 梯度下降法gradient descent algorithm
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    saver = tf.train.Saver()
    tf.reset_default_graph()

    saver.restore(sess, './0_train/graph.ckpt-{}'.format(1000))
    image = sess.run(image)
    result = sess.run(y, feed_dict={x: [image]})
    resultNum = result.argmax()
    print(resultNum)
    return resultNum



cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    ret, frame = cap.read()
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
        print(text)
        cv2.putText(frame, text, (x, y + h + 20), font, 1, (255, 255, 255), 1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


