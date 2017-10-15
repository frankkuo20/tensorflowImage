import glob
paths = glob.glob('imageData/*/*')
print(len(paths))

# import tensorflow as tf
#
# from tensorflow2.cnn import CnnObj
#
# cnnObj = CnnObj()
# x = cnnObj.x
# y_ = cnnObj.y_
# keep_prob = cnnObj.keep_prob
# accuracy = cnnObj.accuracy
#
#
# max_step = 200
#
# tf.reset_default_graph()
# with tf.Session() as sess:
#     print('./cnn_train/graph.ckpt-{}'.format(max_step))
#
#     saver = tf.train.import_meta_graph('./cnn_train/graph.ckpt-{}.meta'.format(max_step))
#     saver.restore(sess, tf.train.latest_checkpoint('./cnn_train/'))
#     sess.run(tf.global_variables_initializer())
#     all_vars = tf.trainable_variables()
#     for v in all_vars:
#         print("%s with value %s" % (v.name, sess.run(v)))
