import tensorflow
import numpt as np
import imageio
import cv2

class configuration():
    def __init__(self, batchsize, data_list, label_list):
        self.n_classes = 101
        self.Height = self.Width = 224
        self.batchsize = batchsize
        self.data_list = data_list
        self.label_list = label_list
        self.query = list(range(len(self.data_list)))
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
            self.query = list(range(len(self.data_list)))
            np.random.shuffle(self.query)
        pick = self.query[:self.batchsize]
        del self.query[:self.batchsize]
        for i in range(self.batchsize):
            index = pick[i]
            [x1[i], x2[i], x3[i]] = self.open_frame(index, is_training=True)
            y[i] = int(label_list[index]) - 1
            
        return x1, x2, x3, y

    def test(self):
        xs = np.zeros((self.batchsize, 250, 224, 224, 3))
        y = np.zeros((self.batchsize), dtype=np.int)
        pick = self.query[:self.batchsize]
        del self.query[:self.batchsize]
        for i in range(self.batchsize):
            index = pick[i]
            xs[i] = self.open_frame(index, is_training=False)
            y[i] = int(label_list[index]) -1
            
        return xs, y
    
    def open_frame(self, index, is_training):
        data = data_list[index]
        nframes = np.load("./frame/" + data.split('.')[0]+ "/nframes.npy") + 1
 
        if is_training:
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

        else:
          w_crop = h_crop = 224
          sample = [(i+1)*(nframes//26) for i in range(25)]
          frame_list = []
          for t in range(25):
            file_dir = "./frame/" + data.split('.')[0] + "/frame_%d.jpg" %sample[t]
            frame = imageio.imread(file_dir)
            for c in range(5):
              for f in range(2):
                frame_ = self.data_augmentation(frame, f, w_crop, h_crop, c)
                frame_list.append(frame_)          
          frames = np.stack(frame_list, axis=0)

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
