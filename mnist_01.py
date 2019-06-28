import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 实现模型
# 
# 创建可操作的变量，x不是一个特定的值，而是一个占位符
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None,10])
# 创建一个权重W和偏置b，全为0的张量初始化
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# y = Wx+b
y = tf.matmul(x, W) + b

# 定义一个损失函数评估模型 交叉熵
# cross_entropy = H_y'(y) = sum(y'log(y))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# 使用GDO以0.01的学习率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 在一个Session里面启动我们的模型，并且初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 开始训练模型，这里我们让模型循环训练1000次！
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评估模型
#
# equal()会给我们一组布尔值[True, False, True, True]
# cast把布尔值转换成浮点数
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()
