import tensorflow as tf
def weight_variable(shape):
    """这个函数给定了全连接神经网络或卷积神经网络网络层的权重，将会使用一个标准差为0.1的截断式正态分布来初始化。"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """这个函数给定了全连接神经网络或卷积神经网络层的偏置量，用常量0.1进行初始化。"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """这个函数指定了我们会常常使用的卷积操作。全卷积（无跳步），其输出大小与输入一致。"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """这个函数设置最大池化为高度和宽度维度的一半的大小，总共是1/4的特征图。"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    """这个函数实际使用的卷积层，偏置数是权重的第三个维度值，也就是输出的特征数。"""
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    """标准全连接层加上偏置，这里没有加上ReLU。"""
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b
