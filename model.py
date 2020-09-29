import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import utils

from tensorflow.keras.layers import (
    Input, Activation, Layer,
    UpSampling2D, Concatenate,
    Add, Conv2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19
from ops import (
    WaveLetPooling, WaveLetUnPooling, Reduction,
    WhiteningAndColoring, get_predict_function)

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
        self.wct = self.build_wct_model()
        # ======= Loss functions ======= #
        recontruct_img = self.wct(img)
        feat = self.encoder(recontruct_img)

        self.trainer = Model(inputs=[img], outputs=[recontruct_img, feat], name="trainer")
        self.trainer.compile(optimizer=Adam(self.lr), loss=["mse",])

        


    def conv_block(self, x, filters, kernel_size,
                    activation='relu', name=""):

        x = Conv2D(filters, kernel_size=kernel_size, strides=1,
                    padding='same', activation=activation, name=name)(x)
        return x


    def build_wct_model(self):
        img = Input(self.img_shape, name='in_img')
        kernel_size = 3
        skips = []

        vgg_model = VGG19(include_top=False,
                         weights='imagenet',
                         input_tensor=Input(self.img_shape),
                         input_shape=self.img_shape)

        vgg_model.trainable = False
        for layer in vgg_model.layers:
            layer.trainable = False

        self.encoder = Model(inputs=vgg_model.inputs,
                            outputs=vgg_model.get_layer('block4_conv1').get_output_at(0),
                            name='encoder')

        # ======= Encoder ======= #
        x = vgg_model.get_layer(VGG_LAYERS[0])(img)
        for layer in VGG_LAYERS[1:]:
            origin_layer = vgg_model.get_layer(layer)
            filters = origin_layer.output_shape[-1]
            x = self.conv_block(x, filters, kernel_size, name=origin_layer.name + "_encode")
            if layer in ['block1_conv2', 'block2_conv2', 'block3_conv4']:
                to_append = [x]
                x, lh, hl, hh= WaveLetPooling(layer)(x)
                to_append += [lh, hl, hh]
                skips.append(to_append)

        # ======= Decoder ======= #
        skip_id = 2
        for layer in VGG_LAYERS[::-1][:-1]:
            filters = vgg_model.get_layer(layer).output_shape[-1]
            name = layer + "_decode"
            if layer in ['block4_conv1', 'block3_conv1', 'block2_conv1']:
                x = self.conv_block(x, filters // 2, kernel_size, name=name)
                original, lh, hl, hh = skips[skip_id]
                x = WaveLetUnPooling(layer)([x, x, x, x, original])
                skip_id -= 1
            else:
                x = self.conv_block(x, filters, kernel_size, name=name)

        out = self.conv_block(x, 3, kernel_size, 'linear', name="output")

        wct = Model(inputs=img, outputs=out, name='wct')

        for layer in wct.layers:
            # dont train waveletpooling layers
            if "_encode" in layer.name:
                name = layer.name.replace("_encode", "")
                layer.set_weights(vgg_model.get_layer(name).get_weights())
                layer.trainable=False

        return wct


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
                feat = self.encoder.predict(content_img)
                loss, *_ = self.trainer.train_on_batch([content_img], [content_img,feat])
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
                gen_img = self.wct.predict(img)
                data_gen.show_imgs(np.concatenate([img, gen_img]))

        self.history = history


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


    def transfer(self, content_img, style_img):
        # step 1.
        content_feat, style_feat = self.en_1([content_img]), self.en_1([style_img])
        content_feat = WhiteningAndColoring()([content_feat, style_feat])
        # step 2.
        content_feat, c_skips = self.pool_1([content_feat])
        style_feat, s_skips = self.pool_1([style_feat])
        content_feat = WhiteningAndColoring()([content_feat, style_feat])
        # step 3.


    def init_transfer_sequence(self):
        # ===== encoder layers ===== #
        self.en_1 = get_predict_function(self.wct, 'block1_conv1')
        self.pool_1 = get_predict_function(self.wct, 'block1_conv2_encode', ['block1_conv2', 'block2_conv1_encode'])
        self.pool_2 = get_predict_function(self.wct, 'block2_conv2_encode', ['block2_conv2', 'block3_conv1_encode'])
        self.pool_3 = get_predict_function(self.wct, 'block3_conv2_encode', ['block3_conv4', 'block4_conv1_decode'])

        # ===== decoder layers ===== #
        self.de_1 = get_predict_function(self.wct, 'block4_conv1_decode')
        self.unpool_1 = get_predict_function(self.wct, 'block4_conv1', 'block3_conv2_decode')
        self.de_2 = get_predict_function(self.wct, 'block3_conv1_decode')
        self.unpool_2 = get_predict_function(self.wct, 'block3_conv1', 'block2_conv2_decode')
        self.de_3 = get_predict_function(self.wct, 'block2_conv1_decode')
        self.unpool_3 = get_predict_function(self.wct, 'block2_conv1', 'block1_conv2_decode')
        self.final = get_predict_function(self.wct, 'output')

        


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