from __future__ import print_function
import sys
sys.path.append("/home/yukaiwang/MTTH/models/slim")
import tensorflow as tf
from nets import inception_v2 as net
import numpy as np 
import os
import imageio
import cv2
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
is_training = True          

label_set = np.genfromtxt("dataset/ucfTrainTestlist/classInd.txt", dtype ='U')
training_set = np.genfromtxt("dataset/ucfTrainTestlist/trainlist01.txt", dtype ='U')
testing_set = np.genfromtxt("dataset/ucfTrainTestlist/testlist01.txt", dtype = 'U')

print("Train")
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
        x = np.zeros((self.batchsize, 224, 224, 3))
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
            x[i] = self.open_frame(index)
            y[i] = int(training_set[index][1]) - 1
        return x, y

    def open_frame(self, index):
        data = training_set[index][0]
        nframes = np.load("./frame/" + data.split('.')[0]+ "/nframes.npy") + 1
        
        flipping = np.random.choice([True, False])
        w_crop = np.random.choice([256, 224, 192, 168])
        h_crop = np.random.choice([256, 224, 192, 168])
        p_crop = np.random.choice(5)

        sample = np.random.choice(nframes)
        file_dir = "./frame/" + data.split('.')[0] + "/frame_%d.jpg" %sample
        frame = imageio.imread(file_dir)
        frame_ = self.data_augmentation(frame, flipping, w_crop, h_crop, p_crop)
        return frame_

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
#batch_x1, batch_x2, batch_x3, batch_y = conf.train_batch()

# define variable    
with tf.name_scope('input_data'):
    with tf.device('/gpu:0'):                   
        x = tf.placeholder(tf.float32, [None,  Height, Width, 3], name = 'x') 
    y = tf.placeholder(tf.int32, [None], name = 'y')
    y_onehot = tf.one_hot(y, n_classes, name = 'label')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

slim = tf.contrib.slim

with tf.device('/gpu:0'):
    with slim.arg_scope(net.inception_v2_arg_scope()) as sc:
        _, end_points = net.inception_v2(x, num_classes = 101,
                                           is_training=is_training, reuse=None)
#print([v.name for v in tf.trainable_variables() if '/Logits/' in v.name])
variables_to_restore = slim.get_variables_to_restore(
                            include=["InceptionV2"], 
                            #exclude=["/Logits/"])
                            exclude=[v.name for v in tf.trainable_variables() if '/Logits/' in v.name])
print([v for v in variables_to_restore if '/Logits/' in v.name])
init_cnn = slim.assign_from_checkpoint_fn('./checkpoints/slim_models/inception_v2.ckpt',
           variables_to_restore)

logits = end_points['Logits']
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
weight_decay = wd*sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = cross_entropy+weight_decay

optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
print(tf.trainable_variables())
var_train = [var for var in tf.trainable_variables() if not 'BatchNorm' in var.name]
print(var_train)
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

    for i in range(240*n_data//128):
        start_time = time.time()
        for j in range(n_minibatches):
            start_time = time.time()
            batch_x, batch_y = conf.train_batch()
            feed_dict = {x:batch_x, y:batch_y, learning_rate: lr}
            _, entropy, accur = sess.run([train_step, cross_entropy, accuracy], feed_dict=feed_dict)
            print('step:%4d-' %i, '{:2d}'.format(j), 
                  'loss'        , '{:8.5f}'.format(entropy), 
                  'accuracy'    , '{:6.4f}'.format(accur),
                  'training time: %ds' % (time.time() - start_time))

        if (i+1) % 500 == 0:
            save_path = saver.save(sess, "./checkpoints/fine_tuned_cnn/inception_v2_SGD_ucfsp1/model_longest.ckpt")
            print('model saved! ', save_path)

        if (i+1) % (90*n_data//128) == 0:
            lr = lr/10.
            print('learing rate decreased to :', lr)
