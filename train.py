from __future__ import print_function
import sys
 
sys.path.append("path/to/slim")
import tensorflow as tf
from nets import inception_v2 as net

import numpy as np 
import os
import time
import conf as 

training_set = np.genfromtxt("dataset/ucfTrainTestlist/trainlist01.txt", dtype ='U')

data_list = training_set[:,0] 
label_list = training_set[:,1]

n_classes = 101
n_data = len(data_list)
batchsize = 64
wd = 5e-4

conf = cf.configuration(batchsize, data_list, label_list)

# define variable    
with tf.name_scope('input_data'):
    with tf.device('/gpu:0'):                   
        x1 = tf.placeholder(tf.float32, [None,  Height, Width, 3], name = 'x1') 
        x2 = tf.placeholder(tf.float32, [None,  Height, Width, 3], name = 'x2') 
        x3 = tf.placeholder(tf.float32, [None,  Height, Width, 3], name = 'x3') 
    y = tf.placeholder(tf.int32, [None], name = 'y')
    y_onehot = tf.one_hot(y, n_classes, name = 'label')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

slim = tf.contrib.slim

# TSN network
with tf.name_scope('model'):
    with tf.device('/gpu:0'):
        with slim.arg_scope(net.inception_v2_arg_scope()) as sc:
            _, end_points_1 = net.inception_v2(x1, num_classes = 101,
                                               is_training=True, reuse=None)
            _, end_points_2 = net.inception_v2(x2, num_classes = 101, 
                                               is_training=True, reuse=True)
            _, end_points_3 = net.inception_v2(x3, num_classes = 101,
                                               is_training=True, reuse=True)
    variables_to_restore = slim.get_variables_to_restore(
                                include=["InceptionV2"], 
                                exclude=[v.name for v in tf.trainable_variables() if '/Logits/' in v.name])

    ## print([v for v in variables_to_restore if '/Logits/' in v.name])
    init_cnn = slim.assign_from_checkpoint_fn('./checkpoints/slim_models/inception_v2.ckpt',
               variables_to_restore)

    logits = (end_points_1['Logits'] + end_points_2['Logits'] + end_points_3['Logits'])/3.

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
    weight_decay = wd*sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss = cross_entropy + weight_decay
    
with tf.name_scope('back_propagation'):   
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    ## print(tf.trainable_variables())
    var_train = [var for var in tf.trainable_variables() if not 'BatchNorm' in var.name]
    grads = optimizer.compute_gradients(loss, var_train)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        capped_grads = [(tf.clip_by_value(grad, -20., 20.), var) for grad, var in grads]
        train_step = optimizer.apply_gradients(capped_grads)

with tf.name_scope('accuracy'):
    pred = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y_onehot,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    init_cnn(sess)

    lr = 0.001
    n_minibatches = int(128/batchsize)
    print("iter is: ", n_minibatches)

    for i in range(80*n_data//128):
        start_time = time.time()
        for j in range(n_minibatches):
            start_time = time.time()
            batch_x1, batch_x2, batch_x3, batch_y = conf.train_batch()
            feed_dict = {x1:batch_x1, x2: batch_x2, x3: batch_x3, y:batch_y, learning_rate: lr}
            _, entropy, accur = sess.run([train_step, cross_entropy, accuracy], feed_dict=feed_dict)
            print('step:%4d-' %i, '{:2d}'.format(j), 
                  'loss'        , '{:8.5f}'.format(entropy), 
                  'accuracy'    , '{:6.4f}'.format(accur),
                  'training time: %ds' % (time.time() - start_time))

        if (i+1) % 500 == 0:
            save_path = saver.save(sess, "pass/to/save/model.ckpt")
            print('model saved! ', save_path)

        if (i+1) % (30*n_data//128) == 0:
            lr = lr/10.
            print('learing rate decreased to :', lr)
