import tensorflow as tf
import input_data

# ===================================== #
#                                       #
#          构建一个多层卷积网络            #
#                                       #
# ===================================== #

# 定义初始值
DATA_DIR = 'MNIST_data/'
NUM_STEPS = 2000
MINIBATCH_SIZE = 50
RATE = 0.01

# 加载MNIST数据
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# 运行TensorFlow的InteractiveSession
# 更加灵活地构建代码
sess = tf.InteractiveSession()

# 为输入图像和目标输出类别创建节点
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 权重初始化
#
# 这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度
# truncated_normal是截断正态，steddev是标准差，取值范围在0.2倍steddev之内   
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化
#
# 卷积层步长stride为1，SAME模版，保证输入输出大小不变
# 池层2*2 max pooling
# strides代表在shape每一维的步长
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                            strides=[1, 2, 2, 1], padding='SAME')

# 为了用卷积层，把x变成一个4D向量，其2，3维代表图片的宽、高，最后一维代表图片的颜色通道数
# input tensor of shape [batch, in height, in width, in channels]
# batch:图片的数量  height:图片的高度   weight:图片的宽度   channal:图片的通道数(彩图:3)
x_image = tf.reshape(x, [-1,28,28,1])

# 第一层卷积
# kernel tensor of shape [filter height, filter width, in channels, out channels]
# 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# 把x_image和全值向量进行卷积，加上偏置项
# 然后应用ReLU激活函数，最后进行max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
# 经过两次卷积，图片尺寸缩小到7*7，连接一个1024的全连接层，用于处理整个图片
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，在输出层之前加入dropout
# 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
# 在训练过程中启用dropout，在测试过程中关闭dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
# 
# 使用ADAM优化器做梯度下降
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 评估模型
correct_production = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_production, 'float'))

# Train
sess.run(tf.global_variables_initializer())
for i in range(NUM_STEPS):
    batch = mnist.train.next_batch(MINIBATCH_SIZE)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step {:d}, training accuracy {:.4}".format(i, train_accuracy))
        
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# Test
test_accuracy = accuracy.eval(feed_dict={
    x:mnist.test.images, y_:mnist.test.labels,keep_prob:1.0})

print("test accuracy %g"%test_accuracy)
    


