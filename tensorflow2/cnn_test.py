import tensorflow as tf

from tensorflow2.cnn import CnnObj
from tensorflow2.constants import input_pipeline
import matplotlib.pyplot as plt
import numpy as np
import math

cnnObj = CnnObj()
x = cnnObj.x
y_ = cnnObj.y_
keep_prob = cnnObj.keep_prob
accuracy = cnnObj.accuracy

h_conv = cnnObj.h_conv
h_conv2 = cnnObj.h_conv2
W_conv = cnnObj.W_conv


test_img_batch, test_label_batch = input_pipeline(["./binData/test.bin"], 200)



# def plotNNFilter(units):
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
#
# def getActivations(layer, stimuli):
#     units = sess.run(layer, feed_dict={x: stimuli, keep_prob: 1.0})
#     plotNNFilter(units)


saver = tf.train.Saver()
max_step = 200

with tf.Session() as sess:
    # 加载模型。模型的文件名称看下本地情况
    print('./cnn_train/graph.ckpt-{}'.format(max_step))
    saver.restore(sess, './cnn_train/graph.ckpt-{}'.format(max_step))

    coord_test = tf.train.Coordinator()
    threads_test = tf.train.start_queue_runners(coord=coord_test)
    test_imgs, test_labels = sess.run([test_img_batch, test_label_batch])

    # imageToUse = test_imgs
    #
    # plt.imshow(np.reshape(imageToUse, [128, 128]), interpolation="nearest", cmap="gray")
    # plt.show()
    # getActivations(h_conv, imageToUse)
    # getActivations(h_conv2, imageToUse)
    # getActivations(W_conv, imageToUse)


    # 预测阶段，keep取值均为1
    acc = sess.run(accuracy, feed_dict={x: test_imgs, y_: test_labels, keep_prob: 1.0})
    print("predict accuracy is %.2f" % acc)
    coord_test.request_stop()
    coord_test.join(threads_test)
