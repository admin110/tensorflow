import tensorflow as tf
import input_data

# 定义初始值
DATA_DIR = 'MNIST_data/'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# 
# 加载数据集
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# 实现模型
# 
# 创建一个可操作的变量，x不是一个特定的值，而是一个占位符
x = tf.placeholder(tf.float32, [None, 784])
# 创建一个权重W和偏置b，全为0的张量初始化
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 定义模型运算
# y = Wx+b
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
# 定义一个损失函数评估模型 交叉熵
# loss = H_y'(y) = -sum(y'log(y))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 使用GDO以0.01的学习率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 评估模型
# 
# 找出预测正确的标签
# 找到标签的索引位置，判断预测值和真实值的索引是否相同，给出一组布尔值[Ture, false,...]
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 将布尔值转化为浮点数，然后取平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练、测试模型
# 
# 让模型训练NUM_STEPS次
# 循环步骤中，随机抓取训练集数据中MINIBATCH_SIZE个批处理数据点，然后用这些数据点作为参数替换占位符
# batch_xs每次获得batch_size大小图片，batch_ys获得标签
with tf.Session() as sess:
    # Train
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = mnist.train.next_batch(MINIBATCH_SIZE)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

    # Test
    ans = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})

# 输出结果
# 
print("Accuracy: {:.4}%".format(ans*100))
