
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
import numpy as np

class WaveLetPooling(Layer):
    def __init__(self, upsample=False):
        super(WaveLetPooling, self).__init__()
        self.upsample = upsample
        square_of_2 = tf.math.sqrt(tf.constant(2, dtype=tf.float32))
        L = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant([[1, 1]], dtype=tf.float32))
        )
        H = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant([[-1, 1]], dtype=tf.float32))
        )

        print(L)
        print(H)

        self.LL = tf.reshape(tf.math.multiply(tf.transpose(L), L), (1, 2, 2, 1))
        self.LH = tf.reshape(tf.math.multiply(tf.transpose(L), H), (1, 2, 2, 1))
        self.HL = tf.reshape(tf.math.multiply(tf.transpose(H), L), (1, 2, 2, 1))
        self.HH = tf.reshape(tf.math.multiply(tf.transpose(H), H), (1, 2, 2, 1))

        print(self.LL)


    def call(self, inputs):
        self.repeat_filters(inputs.shape[-1])
        input_shape = K.int_shape(inputs)
        print(self.LL)
        if self.upsample:
            output_shape = (1, input_shape[1] * 2,
                            input_shape[2] * 2, input_shape[3])
        else:
            output_shape = (1, input_shape[1]//2,
                            input_shape[2]//2, input_shape[3])

        if self.upsample:
            return [tf.nn.conv2d_transpose(inputs, self.LL, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME'),
                    tf.nn.conv2d_transpose(inputs, self.LH, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME'),
                    tf.nn.conv2d_transpose(inputs, self.HL, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME'),
                    tf.nn.conv2d_transpose(inputs, self.HH, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME')]

        return [tf.nn.conv2d(inputs, self.LL, strides=[1, 2, 2, 1], padding='SAME'),
                tf.nn.conv2d(inputs, self.LH, strides=[1, 2, 2, 1], padding='SAME'),
                tf.nn.conv2d(inputs, self.HL, strides=[1, 2, 2, 1], padding='SAME'),
                tf.nn.conv2d(inputs, self.HH, strides=[1, 2, 2, 1], padding='SAME')]


    def compute_output_shape(self, input_shape):
        if self.upsample:
            shape = (input_shape[0], input_shape[1] * 2,
                    input_shape[2] * 2, input_shape[3])

        else:
            shape = (input_shape[0], input_shape[1]//2,
                    input_shape[2]//2, input_shape[3])

        return [shape, shape, shape, shape]

    def repeat_filters(self, repeats):
        self.LL = tf.transpose(tf.repeat(self.LL, repeats, axis=0), (1, 2, 3, 0))
        self.LH = tf.transpose(tf.repeat(self.LH, repeats, axis=0), (1, 2, 3, 0))
        self.HL = tf.transpose(tf.repeat(self.HL, repeats, axis=0), (1, 2, 3, 0))
        self.HH = tf.transpose(tf.repeat(self.HH, repeats, axis=0), (1, 2, 3, 0))


def WhiteningAndColoring(Layer):
    def __init__(self):
        super(WhiteningAndColoring, self).__init__()


    def call(self, inputs):
        """
        Make it works first .
        """
        content, style = inputs
        
        content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
        style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

        Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
        Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

        # CxHxW -> CxH*W
        content_flat = tf.reshape(content_t, (Cc, Hc*Wc))
        style_flat = tf.reshape(style_t, (Cs, Hs*Ws))

        # Content covariance
        mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
        fc = content_flat - mc
        fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc*Wc, tf.float32) - 1.) + tf.eye(Cc)*eps

        # Style covariance
        ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
        fs = style_flat - ms
        fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs*Ws, tf.float32) - 1.) + tf.eye(Cs)*eps

        # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
        with tf.device('/cpu:0'):  
            Sc, Uc, _ = tf.svd(fcfc)
            Ss, Us, _ = tf.svd(fsfs)

        # Filter small singular values
        k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.float32))
        k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.float32))

        # Whiten content feature
        Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
        fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:,:k_c], Dc), Uc[:,:k_c], transpose_b=True), fc)

        # Color content with style
        Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
        fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds), Us[:,:k_s], transpose_b=True), fc_hat)

        # Re-center with mean of style
        fcs_hat = fcs_hat + ms

        # Blend whiten-colored feature with original content feature
        blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)

        # CxH*W -> CxHxW
        blended = tf.reshape(blended, (Cc,Hc,Wc))
        # CxHxW -> 1xHxWxC
        blended = tf.expand_dims(tf.transpose(blended, (1,2,0)), 0)

        return blended

class Reduction(Layer):
    def __init__(self):
        super(Reduction, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs)