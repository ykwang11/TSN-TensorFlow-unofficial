from __future__ import print_function
import sys
 
sys.path.append("path/to/slim")
import tensorflow as tf
from nets import inception_v2 as net

import numpy as np 
import os
import imageio
import cv2
import time

label_set = np.genfromtxt("dataset/ucfTrainTestlist/classInd.txt", dtype ='U')
training_set = np.genfromtxt("dataset/ucfTrainTestlist/trainlist01.txt", dtype ='U')
testing_set = np.genfromtxt("dataset/ucfTrainTestlist/testlist01.txt", dtype = 'U')

data_list = training_set[:,0] 
label_list = training_set[:,1]

n_classes = 101
n_data = len(training_set)
Size = 224
Height = Width = Size
batchsize = 64
wd = 5e-4

class configuration():
    def __init__(self, batchsize, training_set, testing_set):
        self.n_classes = 101
        self.Height = self.Width = 224
        self.batchsize = batchsize
        self.training_set = training_set
        self.testing_set = testing_set
        self.query = list(range(len(self.training_set)))
        np.random.shuffle(self.query)
        self.epoch = 0

    def train_batch(self):
        x1 = np.zeros((self.batchsize, 224, 224, 3))
        x2 = np.zeros((self.batchsize, 224, 224, 3))
        x3 = np.zeros((self.batchsize, 224, 224, 3))
       
        y = np.zeros((self.batchsize), dtype=np.int)
        
        if len(self.query) < self.batchsize:
            self.epoch = self.epoch + 1
            print("")
            print("%d Epoch finished!" % self.epoch)
            print("")
            self.query = list(range(len(self.training_set)))
            np.random.shuffle(self.query)
        pick = self.query[:self.batchsize]
        del self.query[:self.batchsize]

        for i in range(self.batchsize):
            index = pick[i]
            [x1[i], x2[i], x3[i]] = self.open_frame(index)
            y[i] = int(training_set[index][1]) - 1
        return x1, x2, x3, y

    def open_frame(self, index):
        data = training_set[index][0]
        nframes = np.load("./frame/" + data.split('.')[0]+ "/nframes.npy") + 1
        
        flipping = np.random.choice([True, False])
        w_crop = np.random.choice([256, 224, 192, 168])
        h_crop = np.random.choice([256, 224, 192, 168])
        p_crop = np.random.choice(5)

        sample = [np.random.choice(int(nframes/3)), 
                  np.random.choice(int(nframes/3)) + int(nframes/3),
                  np.random.choice(int(nframes/3)) + 2*int(nframes/3)]
        frames = []
        for i in range(3):
            file_dir = "./frame/" + data.split('.')[0] + "/frame_%d.jpg" %sample[i]
            frame = imageio.imread(file_dir)
            frame_ = self.data_augmentation(frame, flipping, w_crop, h_crop, p_crop)
            frames.append(frame_)
        return frames

    def data_augmentation(self, frame, flipping, l1, l2, p):
        frame = cv2.resize(frame, (340, 256), interpolation = cv2.INTER_CUBIC).astype(float)
        ## frame: RGB , mean: BGR = [104, 117, 123]
        frame[...,2] = frame[...,2] - 104.
        frame[...,1] = frame[...,1] - 117.
        frame[...,0] = frame[...,0] - 123.
        Height, Width, Channel = frame.shape
        if flipping:
            frame = cv2.flip(frame,1)
        if p == 0: 
            crop = frame[0:l1, 0:l2]
        if p == 1:            
            crop = frame[0:l1, (Width-l2):Width]
        if p == 2:            
            crop = frame[(Height-l1):Height, 0:l2]
        if p == 3:  
            crop = frame[(Height-l1):Height, (Width-l2):Width]
        if p == 4:            
            crop = frame[int((Height-l1)/2):int((Height+l1)/2), int((Width-l2)/2):int((Width+l2)/2)]
        frame = cv2.resize(crop, (224, 224), interpolation = cv2.INTER_CUBIC)
        frame = frame/255.
        return frame

conf = configuration(batchsize, training_set, testing_set)

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
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
weight_decay = wd*sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = cross_entropy+weight_decay

optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
## print(tf.trainable_variables())
var_train = [var for var in tf.trainable_variables() if not 'BatchNorm' in var.name]
grads = optimizer.compute_gradients(loss, var_train)

with tf.name_scope('back_propagation'):
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
