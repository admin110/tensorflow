import auxiliary as aux
import input_data
import tensorflow as tf
import numpy as np

# ===================================== #
#                                       #
#          使用自定义辅助函数实例           #
#                                       #
# ===================================== #

# 定义初始值 
# 
DATA_DIR = 'MNIST_data/'
STEPS = 1000
BATCH_SIZE = 50

# 编写模型
#
# 为输入图像和目标输出类别创建节点
with tf.name_scope('build_mod') as scope:
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # 构建卷积层
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = aux.conv_layer(x_image, shape=[5, 5, 1, 32])
    conv1_pool = aux.max_pool_2x2(conv1)

    conv2 = aux.conv_layer(conv1_pool, [5, 5, 32, 64])
    conv2_pool = aux.max_pool_2x2(conv2)

    conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
    full1 = tf.nn.relu(aux.full_layer(conv2_flat, 1024))

    keep_prob = tf.placeholder(tf.float32)
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

    y_conv = aux.full_layer(full1_drop, 10)

    # 训练模型计算图，使用梯度下降
    # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 评估模型计算图
    correct_production = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_production, tf.float32))

# 训练模型
#
# 使用mnist数据集
with tf.name_scope('train_mod') as scope:
    # 引用数据集
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

    # 在session里面启动模型
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            batch = mnist.train.next_batch(BATCH_SIZE)

            if i%100 ==0:
                train_accuracy = sess.run(accuracy, feed_dict={
                                                        x: batch[0], 
                                                        y_: batch[1], 
                                                        keep_prob: 1.0})
                print('step {}, training_accuracy {}'.format(i, train_accuracy))
            
            sess.run(train_step, feed_dict={
                                    x: batch[0], 
                                    y_: batch[1], 
                                    keep_prob: 0.5})
        # Test
        # test_accuracy = sess.run(accuracy, feed_dict={
        #                                         x: mnist.test.images, 
        #                                         y_: mnist.test.labels, 
        #                                         keep_prob:1.0})
        
        X = mnist.test.images.reshape(10, 1000, 784)
        Y = mnist.test.labels.reshape(10, 1000, 10)
        test_accuracy = np.mean([sess.run(accuracy,
                                feed_dict={x: X[i], y_: Y[i], keep_prob:1.0})
                                for i in range(10)])
        print("test accuracy: {}".format(test_accuracy))
            
