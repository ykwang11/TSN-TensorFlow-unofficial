from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np 
import os
import time
import imageio
import cv2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean('is_training', True, 'Extract the training data or not')
print('Extracting the training data', FLAGS.is_training)

label_set = np.genfromtxt("dataset/ucfTrainTestlist/classInd.txt", dtype ='U')
training_set = np.genfromtxt("dataset/ucfTrainTestlist/trainlist01.txt", dtype ='U')
testing_set = np.genfromtxt("dataset/ucfTrainTestlist/testlist01.txt", dtype = 'U')


if FLAGS.is_training:
    data_set = training_set[:,0]
else:
    data_set = testing_set
print(data_set)

Height = 256
Width = 340

def make_sure_path_exists(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

for data in data_set:
    frame_dir = "./frame/" + data.split('.')[0]
    make_sure_path_exists(frame_dir)
    
    file_dir = "UCF-101/" + data
    with imageio.get_reader(file_dir,  'ffmpeg') as vid:
        nframes = vid.get_meta_data()['nframes']
        try:
            for i, frame in enumerate(vid):
                ## note that the order of WH is different between opencv and imageio
                n_frames = i
                frame = cv2.resize(frame, (Width, Height), interpolation = cv2.INTER_CUBIC)
                imageio.imwrite(frame_dir+'/frame_%d.jpg' %i, frame) 
        except:
            print('runtime error:', data)
            print(n_frames)
        np.save(frame_dir + '/nframes.npy', n_frames)
