import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

number_of_points = 200
x_point = []
y_point = []
a = 0.22
b = 0.78

for i in range(number_of_points):
    x = np.random.normal(0.0, 0.5)
    y = a * x + b + np.random.normal(0.0, 0.1)
    x_point.append([x])
    y_point.append([y])

# plt.plot(x_point, y_point, 'o', label='Input data')
# plt.legend()
# plt.show()

A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))
y = A * x_point + B
cost_function = tf.reduce_mean(tf.square(y - y_point))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost_function)
init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_point, y_point)
plt.ion()
plt.show()

with tf.Session() as sess:
    sess.run(init)
    for step in range(0, 31):
        sess.run(train)
        if (step % 5) == 0:
            # plt.plot(x_point, y_point, 'o', label='step = {}'.format(step))

            # plt.plot(x_point, sess.run(A)*x_point+sess.run(B))
            # plt.legend()
            # plt.show()
            lines = ax.plot(x_point, (sess.run(A) * x_point + sess.run(B)), 'r-', lw=3)
            plt.pause(1)

            print(step, sess.run(A), sess.run(B))

    plt.pause(5)
