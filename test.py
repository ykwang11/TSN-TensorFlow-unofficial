from __future__ import print_function
import sys
sys.path.append("/home/yukaiwang/MTTH/models/slim")
import tensorflow as tf
from nets import inception_v2 as net
import numpy as np 
import time
import conf as cf

label_set = np.genfromtxt("dataset/ucfTrainTestlist/classInd.txt", dtype ='U')
testing_set = np.genfromtxt("dataset/ucfTrainTestlist/testlist01.txt", dtype = 'U')

print("Test")
data_list = testing_set
label_list = np.zeros(len(testing_set))
for index, data in enumerate(data_list):
    label_name = data.split('/')[0]
    label_index = np.where(label_set==label_name)[0][0]
    label = label_set[label_index][0]
    label_list[index] = label

n_classes = 101
n_test_data = len(testing_set)
Size = 224
Height = Width = Size
batchsize = 1

conf = cf.configuration(batchsize, data_list, label_list)

# define variable    
with tf.name_scope('input_data'):
    x = tf.placeholder(tf.float32, [None,  Height, Width, 3], name = 'x1') 
    y = tf.placeholder(tf.int32, [None], name = 'y')
    y_onehot = tf.one_hot(y, n_classes, name = 'label')

slim = tf.contrib.slim
with slim.arg_scope(net.inception_v2_arg_scope()) as sc:    
    _, end_points = net.inception_v2(x, num_classes = 101,
                                     is_training=False, reuse=None)
init_cnn = slim.assign_from_checkpoint_fn(
    'pass/to/save/model.ckpt',
    slim.get_model_variables())

logits = end_points['Logits']

with tf.name_scope('accuracy'):
    pred = tf.nn.softmax(logits,1)
    pred_restore = tf.placeholder(pred.dtype, pred.shape)
    correct_pred = tf.equal(tf.argmax(pred_restore,1), tf.argmax(y_onehot,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# run
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    init_cnn(sess)

    n_minibatches = n_test_data//batchsize
    print("iter is: ", n_minibatches)

    accur_sum = 0.0
    for i in range(n_minibatches):
      start_time = time.time()
      batch_xs, batch_y = conf.test()
      pred_ = np.zeros((batchsize, n_classes))
      for j in range(250):
        feed_dict = {x: batch_xs[:, j, ...], y: batch_y}
        pred_ += sess.run(pred, feed_dict=feed_dict)
      accur = sess.run(accuracy, feed_dict={pred_restore: pred_, y: batch_y})
      if (i+1) % 50 == 0:
        print(accur, time.time() - start_time)
      accur_sum = accur_sum + accur
    print('final accuracy is', accur_sum/n_minibatches)

