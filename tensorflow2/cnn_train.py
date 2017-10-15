from tensorflow2.cnn import CnnObj
from tensorflow2.constants import input_pipeline
import tensorflow as tf


cnnObj = CnnObj()
init = cnnObj.init
x = cnnObj.x
y_ = cnnObj.y_
init = cnnObj.init
keep_prob = cnnObj.keep_prob
train_step = cnnObj.train_step
accuracy = cnnObj.accuracy

img_batch, label_batch = input_pipeline(["./binData/train.bin"], 30)

disp_step = 5
save_step = 100
max_step = 200  # 最大迭代次数
step = 0
saver = tf.train.Saver()  # 用来保存模型的
epoch = 5



with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(init)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop() and step < max_step:
            step += 1
            imgs, labels = sess.run([img_batch, label_batch])

            # 训练。训练时dropout层要有值。
            sess.run(train_step, feed_dict={x: imgs, y_: labels, keep_prob: 0.5})
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
