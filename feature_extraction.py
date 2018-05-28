from __future__ import print_function
import sys
sys.path.append("/home/yukaiwang/MTTH/models/slim")
import tensorflow as tf
from nets import inception_v2 as net
import numpy as np 
import imageio
import cv2
import time

is_training = False
is_flipping = True

label_set = np.genfromtxt("dataset/ucfTrainTestlist/classInd.txt", dtype ='U')
training_set = np.genfromtxt("dataset/ucfTrainTestlist/trainlist01.txt", dtype ='U')
testing_set = np.genfromtxt("dataset/ucfTrainTestlist/testlist01.txt", dtype = 'U')


def make_sure_path_exists(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

if is_training is True:
    print("Train!")
    data_list = training_set[:,0]
    label_list = training_set[:,1]
else:
    print("Test!")
    data_list = testing_set
    label_list = np.zeros((len(testing_set)))
    for index, data in enumerate(data_list):
        label_name = data.split('/')[0]
        label_index = np.where(label_set==label_name)[0][0]
        label = label_set[label_index][0]
        #print(label)
        label_list[index] = label


n_classes = len(label_set)
n_data = len(training_set)
Size = 224
Height = Width = Size

# define variable                           
xs = tf.placeholder(tf.float32, [None,  Height, Width, 3]) 

slim = tf.contrib.slim
with slim.arg_scope(net.inception_v2_arg_scope()) as sc:
    logits, end_points = net.inception_v2(xs, num_classes = 101, is_training=False)
init_cnn = slim.assign_from_checkpoint_fn('path/to/save/model.ckpt', slim.get_model_variables())

pool = tf.get_default_graph().get_tensor_by_name("InceptionV2/Logits/AvgPool_1a_7x7/AvgPool:0")
feat = tf.squeeze(pool, [1, 2], name='SpatialSqueeze')


def open_video(data, is_flipping):
    file_dir = "dataset/UCF-101/" + data
    with imageio.get_reader(file_dir,  'ffmpeg') as vid:
        nframes = vid.get_meta_data()['nframes']
        xs = [np.zeros((nframes, Height, Width, 3)) for p in range(5)]
        try:
            for i, frame in enumerate(vid):
                for p in range(5)
                    xs[p][i] = data_augmentation(frame, flipping = is_flipping, p = p)                
        except:
            print('runtime error:', data)
    return xs, nframes

def data_augmentation(frame, flipping = False, l1 = 224, l2 = 224, p = 4):
    frame = cv2.resize(frame, (340, 256), interpolation = cv2.INTER_CUBIC).astype(float)
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
        crop = frame[(Height-l1)/2:(Height+l1)/2, (Width-l2)/2:(Width+l2)/2]
    if p == 5:            
        crop = frame
    frame = cv2.resize(crop, (224, 224), interpolation = cv2.INTER_CUBIC) 
    frame = frame/255.
    return frame

## run
with tf.Session() as sess:

    init_cnn(sess)
    feat_dict = {}
    print(data_list.shape)    
    i = 0
    for index, data in enumerate(data_list):
        start_time = time.time()
        print("/n" + data)

        npy_dir = []        
        if is_flipping is True:
            npy_dir.append('./features/ucfsp1/upperleft_rgb_rescaledi_flipped/' + data.split('/')[0])
            npy_dir.append('./features/ucfsp1/upperright_rgb_rescaled_flipped/' + data.split('/')[0])
            npy_dir.append('./features/ucfsp1/lowerleft_rgb_rescaled_flipped/' + data.split('/')[0])
            npy_dir.append('./features/ucfsp1/lowerright_rgb_rescaled_flipped/' + data.split('/')[0])
            npy_dir.append('./features/ucfsp1/centorcrop_rgb_rescaled_flipped/' + data.split('/')[0])
        else:
            npy_dir.append('./features/ucfsp1/upperleft_rgb_rescaled/' + data.split('/')[0])
            npy_dir.append('./features/ucfsp1/upperright_rgb_rescaled/' + data.split('/')[0])
            npy_dir.append('./features/ucfsp1/lowerleft_rgb_rescaled/' + data.split('/')[0])
            npy_dir.append('./features/ucfsp1/lowerright_rgb_rescaled/' + data.split('/')[0])
            npy_dir.append('./features/ucfsp1/centorcrop_rgb_rescaled/' + data.split('/')[0])

        npy_path = []
        for p in range(5):
            npy_path.append(npy_dir[p] + '/' + data.split('/')[1].split('.')[0] + '.npy')
            make_sure_path_exists(npy_dir[p])

        if os.path.isfile(npy_path[0]):
            print('the file has already existed')
        else:
            batches, nframes = open_video(data, is_flipping)
            # in order to reduce the GPU memory usage and prevent OOM
            if nframes > 700:
                seg_leng = nframes/3                
                features = [np.zeros((3, seg_leng, 1024)) for i in range(5)]
                for p in range(5):
                    xs_segs = []
                    for seg in xrange(3):
                        xs_segs.append(batches[p][seg*seg_leng:(seg+1)*seg_leng])
                        features[p][seg] = sess.run(feat, feed_dict = {xs: xs_seg1})
                    features[p] = np.reshape(features[p], (-1, 1024))

            else:
                features = []
                for p in range(5):
                    features.append(sess.run(feat, feed_dict = {xs: batches[p]}))

            for p in range(5):
                np.save(npy_path[p], features[p])
            print(("file saved: ", npy_path[0]))
