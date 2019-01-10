import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 学习率（切线步长）
learning_rate = 0.01
# 训练次数
training_epochs = 50

points_num=100

x_data = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
y_data = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

# 绘制点阵
plt.plot(x_data, y_data, 'r*', label='Original data')
plt.legend() # 展示标签
plt.show() # 展示图

# tf Graph Input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 设置模型权重
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")
# 构建线性模型 y=wx+b
y = W * x_data + b
# 损失函数 (y-y_data)^2/n
cost = tf.reduce_mean(tf.square(y-y_data))
# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# 开始训练
with tf.Session() as sess:
    # 执行初始化操作
    sess.run(init)
    # 拟合模型数据
    for epoch in range(training_epochs):
        for (x, y) in zip(x_data, y_data):
            sess.run(optimizer, feed_dict={X:x, Y:y})
        training_cost = sess.run(cost, feed_dict={X: x_data, Y: y_data})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    print(x_data, sess.run(W))
    plt.plot(x_data, y_data, "r*", label='Fitted point')
    # 画出拟合线
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()