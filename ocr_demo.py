# -*- coding:utf-8 -*-
'''
钢印识别demo
'''


import tensorflow as tf
import data_loader
import numpy as np
import tensorflow.contrib.slim as slim
import img_preprocess


img_preprocess.img_pro()

train = False  #进行训练还是测试

max_steps = 410
learning_rate = 0.000075
dropout = 0.8
root_dir = './data/datasets'
dataset = 'dset1'
batch_size = 4
n_label = 4
log_dir = './data/datasets/dset1/log'
dict = {0:'0', 1:'7', 2:'F', 3:'N'}


#训练集
train_set = data_loader.DataSet(root_dir, dataset, 'train',
                                batch_size,n_label,
                                data_aug=False, shuffle=True)
#测试集
test_set = data_loader.DataSet(root_dir, dataset, 'test',
                               batch_size, n_label,
                               data_aug=False, shuffle=False)


sess = tf.InteractiveSession()


with tf.name_scope('input'):
    #训练样本和标签的placeholder
    x = tf.placeholder(tf.float32, [batch_size, 48, 48, 3], name='x-input')
    y_ = tf.placeholder(tf.float32, [batch_size, n_label], name='y-input')


with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 48, 48, 3])
    tf.summary.image('input', image_shaped_input, n_label)


def weight_variable(shape):
    #生成权重的函数
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    #生成偏差的函数
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    #变量汇总函数
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#  全连接层函数
def nn_layer(input_tensor, input_dim, output_dim, layer_name,
             act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

def feed_dict(train):
    if train:
        xs, ys = train_set.next_batch()
        k = dropout
    else:
        xs, ys = test_set.xs, test_set.ys
        xs = np.array(xs)
        ys = np.array(ys)
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

with tf.name_scope('conv'):
    #卷积层，有3层
    with tf.name_scope('conv1'):
        net = slim.conv2d(x, 64, kernel_size=3, stride=1, padding='SAME')
        net = slim.batch_norm(net)
        net = tf.nn.relu(net)
        net = slim.max_pool2d(net, kernel_size=3)
        net = slim.dropout(net, keep_prob=1)

    with tf.name_scope('conv2'):
        net = slim.conv2d(net, 128, kernel_size=3, stride=1, padding='SAME')
        net = slim.batch_norm(net)
        net = tf.nn.relu(net)
        net = slim.max_pool2d(net, kernel_size=3)
        net = slim.dropout(net, keep_prob=1)

    with tf.name_scope('conv3'):
        net = slim.conv2d(net, 256, kernel_size=3, stride=1, padding='SAME')
        net = slim.batch_norm(net)
        net = tf.nn.relu(net)
        net = slim.max_pool2d(net, kernel_size=3)
        net = slim.dropout(net, keep_prob=1)

reshape = tf.reshape(net, [batch_size, -1])
dim = reshape.get_shape()[1].value
#print (reshape.get_shape())

#全连接层1
hidden1 = nn_layer(reshape, dim, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

#全连接层2，输出层
y = nn_layer(dropped, 500, n_label, 'layer2', act=tf.identity)

#计算交叉熵损失
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

#梯度下降
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#计算准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)


merged = tf.summary.merge_all()
if train:
    #训练过程
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    tf.global_variables_initializer().run()


    saver = tf.train.Saver()
    for i in range(max_steps):
        if i % 10 == 0:
            summary, acc, loss = sess.run([merged, accuracy, cross_entropy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print ( 'Accuracy at step %s : %s loss=%s'%(i, acc, loss))
        else:
            if i % 101 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                      options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d'%i)
                train_writer.add_summary(summary, i)
                saver.save(sess, log_dir + '/model.ckpt', i)
                print ('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


else:
    #测试过程
    module_file = tf.train.latest_checkpoint(log_dir)
    #print (module_file)
    saver = tf.train.Saver()
    saver.restore(sess,module_file)
    predict = sess.run([y], feed_dict=feed_dict(False))
    predict = list(np.argmax(predict, 1)[0])

    result = []
    #print (predict)
    for i in range(len(predict)):
         result.append(dict[predict[i]])
    print (result)




