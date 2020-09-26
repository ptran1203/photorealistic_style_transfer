
from keras.layers import Layer
from tensorflow.nn import conv2d, conv2d_transpose
import numpy as np

class WaveLetPooling(Layer):
    def __init__(self, upsample=False):
        super(WaveLetPooling, self).__init__()
        self.upsample = upsample
        L = 1 / np.sqrt(2) * np.array([[1, 1]], dtype=np.float32)
        H = 1 / np.sqrt(2) * np.array([[-1, 1]], dtype=np.float32)

        self.LL = (np.transpose(L) * L).reshape((1, 2, 2, 1))
        self.LH = (np.transpose(L) * H).reshape((1, 2, 2, 1))
        self.HL = (np.transpose(H) * L).reshape((1, 2, 2, 1))
        self.HH = (np.transpose(H) * H).reshape((1, 2, 2, 1))


    def call(self, inputs):
        self.repeat_filters(inputs.shape[-1])
        print(self.LL.shape)
        if self.upsample:
            return (conv2d_transpose(inputs, self.LL, strides=[1, 2, 2, 1], padding='SAME'),
                    conv2d_transpose(inputs, self.LH, strides=[1, 2, 2, 1], padding='SAME'),
                    conv2d_transpose(inputs, self.HL, strides=[1, 2, 2, 1], padding='SAME'),
                    conv2d_transpose(inputs, self.HH, strides=[1, 2, 2, 1], padding='SAME'))


        return (conv2d(inputs, self.LL, strides=[1, 2, 2, 1], padding='SAME'),
                conv2d(inputs, self.LH, strides=[1, 2, 2, 1], padding='SAME'),
                conv2d(inputs, self.HL, strides=[1, 2, 2, 1], padding='SAME'),
                conv2d(inputs, self.HH, strides=[1, 2, 2, 1], padding='SAME'))


    def repeat_filters(self, repeats):
        self.LL = np.transpose(np.repeat(self.LL, repeats, axis=0), (1, 2, 3, 0))
        self.LH = np.transpose(np.repeat(self.LH, repeats, axis=0), (1, 2, 3, 0))
        self.HL = np.transpose(np.repeat(self.HL, repeats, axis=0), (1, 2, 3, 0))
        self.HH = np.transpose(np.repeat(self.HH, repeats, axis=0), (1, 2, 3, 0))
