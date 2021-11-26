import tensorflow as tf
import numpy as np
w1 = tf.Variable(tf.random_normal([2, 3], stddev = 10, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 10, seed = 1))
x = tf.placeholder(tf.float32, shape = (1, 2), name = "input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
temp1 = np.random.normal(size = 2)
temp2 = temp1.reshape(1, 2)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y, feed_dict = {x: temp2}))