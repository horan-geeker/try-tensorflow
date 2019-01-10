import tensorflow as tf

const1 = tf.constant([[2,2]])
const2 = tf.constant([[4],
                      [4]])
multiple = tf.matmul(const1, const2)
print(multiple)
with tf.Session() as sess:
    print(sess.run(multiple))