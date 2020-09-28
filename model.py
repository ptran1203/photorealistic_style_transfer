import tensorflow as tf
import keras
import numpy as np
import datetime
import matplotlib.pyplot as plt
import utils
import keras.backend as K

from keras.layers.convolutional import Conv2D
from keras.layers import Input, Activation, Layer, UpSampling2D, Concatenate, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from ops import WaveLetPooling, Reduction, WhiteningAndColoring

try:
    # In case run on google colab
    from google.colab.patches import cv2_imshow
except ImportError:
    from cv2 import imshow as cv2_imshow


VGG_LAYERS = [
    'block1_conv1', 'block1_conv2',
    'block2_conv1', 'block2_conv2',
    'block3_conv1', 'block3_conv2',
    'block3_conv3', 'block3_conv4',
    'block4_conv1',
]

class WCT2:
    def __init__(self, base_dir, rst, lr,
                show_interval=25,
                style_loss_weight=1):
        self.base_dir = base_dir
        self.rst = rst
        self.lr = lr
        self.show_interval = show_interval
        self.img_shape = (self.rst, self.rst, 3)
        img = Input(self.img_shape)

        self.build_wct_model()

        self.encoder = Model(inputs=self.wct.inputs,
                             outputs=self.wct.get_layer('block4_conv1').get_output_at(0),
                             name='encoder')
        self.encoder.summary()

        # ======= Loss functions ======= #
        recontruct_img = self.wct(img)
        gen_feat = self.encoder(recontruct_img)

        self.trainer = Model(inputs=img, outputs=[recontruct_img, gen_feat], name="trainer")
        self.trainer.compile(optimizers=Adam(self.lr), loss=["mse", "mse"])
    def conv_block(self, x, filters, kernel_size,
                    activation='relu'):

        x = Conv2D(filters, kernel_size=kernel_size, strides=1,
                    padding='same', activation=activation)(x)
        return x


    def build_wct_model(self):
        images = Input(self.img_shape)
        kernel_size = 3
        skips = []

        vgg_model = VGG19(include_top=False,
                         weights='imagenet',
                         input_tensor=Input(self.img_shape),
                         input_shape=self.img_shape)

        vgg_model.trainable = False
        for layer in vgg_model.layers:
            layer.trainable = False

        img = Input(self.img_shape)

        # ======= Encoder ======= #
        x = vgg_model.get_layer(VGG_LAYERS[0])(img)
        for layer in VGG_LAYERS[1:]:
            x = vgg_model.get_layer(layer)(x)

            if layer in ['block1_conv2', 'block2_conv2', 'block3_conv4']:
                x, *skip= WaveLetPooling()(x)
                skips.append(Concatenate()(skip))

        # ======= Decoder ======= #
        skip_id = 2
        for layer in VGG_LAYERS[::-1][:-1]:
            x = self.conv_block(
                x,
                vgg_model.get_layer(layer).output_shape[-1],
                kernel_size)

            if layer in ['block4_conv1', 'block3_conv1', 'block2_conv1']:
                x = WaveLetPooling(upsample=True)(x)
                x = Add()([x, skips[skip_id]])
                skip_id -= 1

        out = self.conv_block(x, 3, kernel_size, 'linear')

        self.wct = Model(inputs=img, outputs=out, name='wct')
        self.wct.summary()


    @staticmethod
    def init_hist():
        return {
            "loss": [],
            "val_loss": []
        }


    def train(self, data_gen, epochs):
        history = self.init_hist()
        print("Train on {} samples".format(len(data_gen.x)))

        for e in range(epochs):
            start_time = datetime.datetime.now()
            print("Train epochs {}/{} - ".format(e + 1, epochs), end="")

            batch_loss = self.init_hist()
            for content_img in data_gen.next_batch():
                loss = self.wct.train_on_batch(content_img, content_img)
                batch_loss['loss'].append(loss)

            # evaluate
            # batch_loss['val_loss'] = 

            mean_loss = np.mean(np.array(batch_loss['loss']))
            mean_val_loss = 0#np.mean(np.array(batch_loss['val_loss']))

            history['loss'].append(mean_loss)
            history['val_loss'].append(mean_val_loss)

            print("Loss: {}, Val Loss: {} - {}".format(
                mean_loss, mean_val_loss,
                datetime.datetime.now() - start_time
            ))

            if e % self.show_interval == 0:
                # self.save_weight()
                idx = np.random.randint(0, data_gen.max_size - 1)
                img = data_gen.x[idx:idx+1]
                gen_img = self.generate(img)
                data_gen.show_imgs(np.concatenate([img, gen_img]))

        self.history = history
        return history


    def plot_history(self):
        plt.plot(self.history['loss'], label='train loss')
        plt.plot(self.history['val_loss'], label='val loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('Segmentation model')
        plt.legend()
        plt.show()


    def save_weight(self):
        try:
            self.wct.save_weights(self.base_dir + '/wct2.h5')
        except Exception as e:
            print("Could not load model, {}".format(str(e))) 


    def load_weight(self):
        try:
            self.wct.load_weights(self.base_dir + '/wct2.h5')
        except Exception as e:
            print("Save model failed, {}".format(str(e))) 


    def generate(self, content_imgs, style_imgs):
        return self.wct.predict([content_imgs, style_imgs])


    def show_sample(self, content_img, style_img,
                    concate=True, denorm=True, deprocess=True):
        gen_img = self.generate(content_img, style_img)

        if concate:
            return utils.show_images(np.concatenate([content_img, style_img, gen_img]), denorm, deprocess)

        if denorm:
            content_img = utils.de_norm(content_img)
            style_img = utils.de_norm(style_img)
            gen_img = utils.de_norm(gen_img)
        if deprocess:
            content_img = utils.deprocess(content_img)
            style_img = utils.deprocess(style_img)
            gen_img = utils.deprocess(gen_img)

        cv2_imshow(content_img[0])
        cv2_imshow(style_img[0])
        cv2_imshow(gen_img[0])