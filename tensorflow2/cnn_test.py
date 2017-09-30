import tensorflow as tf
from tensorflow2.constants import input_pipeline
from tensorflow2.cnn import x, y_, keep_prob, accuracy


test_img_batch, test_label_batch = input_pipeline(["./binData/test.bin"], 50)

saver = tf.train.Saver()
max_step = 200

with tf.Session() as sess:
    # 加载模型。模型的文件名称看下本地情况
    print('./cnn_train/graph.ckpt-{}'.format(max_step))
    saver.restore(sess, './cnn_train/graph.ckpt-{}'.format(max_step))

    coord_test = tf.train.Coordinator()
    threads_test = tf.train.start_queue_runners(coord=coord_test)
    test_imgs, test_labels = sess.run([test_img_batch, test_label_batch])
    # 预测阶段，keep取值均为1
    acc = sess.run(accuracy, feed_dict={x: test_imgs, y_: test_labels, keep_prob: 1.0})
    print("predict accuracy is %.2f" % acc)
    coord_test.request_stop()
    coord_test.join(threads_test)
