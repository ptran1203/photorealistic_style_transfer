import tensorflow as tf
import keras
import numpy as np
import datetime
import matplotlib.pyplot as plt
import utils
import keras.backend as K

from keras.layers.convolutional import Conv2D
from keras.layers import Input, Activation, Layer, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

try:
    # In case run on google colab
    from google.colab.patches import cv2_imshow
except ImportError:
    from cv2 import imshow as cv2_imshow

DEFAULT_STYLE_LAYERS = [
    'block1_conv1', 'block2_conv1',
    'block3_conv1', 'block4_conv1',
]
DEFAULT_LAST_LAYER = 'block4_conv1'


class Reduction(Layer):
    def __init__(self):
        super(Reduction, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs)

class WCT2:
    def __init__(self, base_dir, rst, lr,
                style_layer_names=DEFAULT_STYLE_LAYERS,
                last_layer=DEFAULT_LAST_LAYER,
                show_interval=25,
                style_loss_weight=1,
                pre_trained_model='vgg16'):
        self.base_dir = base_dir
        self.rst = rst
        self.pre_trained_model = pre_trained_model
        self.lr = lr
        self.style_layer_names = style_layer_names
        self.last_layer = last_layer
        self.show_interval = show_interval
        img_shape = (self.rst, self.rst, 3)

        # ===== Build the model ===== #
        self.encoder = self.build_encoder()
        self.style_layers = self.build_style_layers()
        content_img = Input(shape=img_shape)
        style_img = Input(shape=img_shape)

        content_feat = self.encoder(content_img)
        style_feat = self.encoder(style_img)

        combined_feat = AdaptiveInstanceNorm()([content_feat, style_feat])
        self.init_rst = K.int_shape(combined_feat)[1]
        self.decoder = self.build_decoder((self.init_rst, self.init_rst, 512))

        gen_img = self.decoder(combined_feat)
        gen_feat = self.encoder(gen_img)

        self.transfer_model = Model(inputs=[content_img, style_img],
                                    outputs=gen_img)
        content_loss = K.mean(K.square(combined_feat - gen_feat), axis=[1, 2])
        self.transfer_model.add_loss(Reduction()(content_loss))
        self.transfer_model.add_loss(style_loss_weight*self.compute_style_loss(gen_img, style_img))
        self.transfer_model.compile(optimizer=Adam(self.lr),
                                    loss=["mse"],
                                    loss_weights=[0.0])


    def compute_style_loss(self, gen_img, style_img):
        gen_feats = self.style_layers(gen_img)
        style_feats = self.style_layers(style_img)
        style_loss = []
        axis = [1, 2]
        for i in range(len(style_feats)):
            gmean = K.mean(gen_feats[i], axis=axis)
            gstd = K.std(gen_feats[i], axis=axis)

            smean = K.mean(style_feats[i], axis=axis)
            sstd = K.std(style_feats[i], axis=axis)

            style_loss.append(
                K.sum(K.square(gmean - smean)) +
                K.sum(K.square(gstd - sstd))
            )

        return Reduction()(style_loss)


    def build_style_layers(self):
        return Model(
            inputs=self.encoder.inputs,
            outputs=[self.encoder.get_layer(l).get_output_at(0) \
                for l in self.style_layer_names]
        )


    def build_encoder(self):
        input_shape = (self.rst, self.rst, 3)
        vggnet = VGG16 if self.pre_trained_model == 'vgg16' else VGG19
        model = vggnet(
            include_top=False,
            weights='imagenet',
            input_tensor=Input(input_shape),
            input_shape=input_shape,
        )
        print('Encoder: {}'.format(model.name))
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False

        return Model(
            inputs=model.inputs,
            outputs=model.get_layer(self.last_layer).get_output_at(0)
        )


    def conv_block(self, x, filters, kernel_size,
                    activation='relu', up_sampling=False):

        x = Conv2D(filters, kernel_size=kernel_size, strides=1,
                    padding='same', activation=activation)(x)

        if up_sampling:
            x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        return x


    def build_decoder(self, input_shape):
        feat = Input(input_shape)
        kernel_size = 3

        x = self.conv_block(feat, 512, kernel_size=kernel_size, up_sampling=True)

        x = self.conv_block(x, 256, kernel_size=kernel_size)
        x = self.conv_block(x, 256, kernel_size=kernel_size)
        x = self.conv_block(x, 256, kernel_size=kernel_size)
        x = self.conv_block(x, 256, kernel_size=kernel_size, up_sampling=True)

        # x = self.conv_block(x, 128, kernel_size=kernel_size)
        # x = self.conv_block(x, 128, kernel_size=kernel_size)
        x = self.conv_block(x, 128, kernel_size=kernel_size)
        x = self.conv_block(x, 128, kernel_size=kernel_size, up_sampling=True)

        x = self.conv_block(x, 64, kernel_size=kernel_size)
        x = self.conv_block(x, 64, kernel_size=kernel_size)

        style_image = self.conv_block(x, 3, kernel_size=kernel_size, activation='linear')

        model = Model(inputs=feat, outputs=style_image, name='decoder')
        return model


    @staticmethod
    def init_hist():
        return {
            "loss": [],
            "val_loss": []
        }


    def train(self, data_gen, epochs, augment_factor=0):
        history = self.init_hist()
        print("Train on {} samples".format(len(data_gen.x)))

        for e in range(epochs):
            start_time = datetime.datetime.now()
            print("Train epochs {}/{} - ".format(e + 1, epochs), end="")

            batch_loss = self.init_hist()
            for content_img, style_img in data_gen.next_batch(augment_factor):
                loss = self.transfer_model.train_on_batch([content_img, style_img],
                                                          style_img)
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
                self.save_weight()
                idx = np.random.randint(0, data_gen.max_size - 1)
                cimg, simg = data_gen.x[idx:idx+1], data_gen.y[idx:idx+1]
                gen_img = self.generate(cimg, simg)
                data_gen.show_imgs(np.concatenate([cimg, simg, gen_img]))

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
            self.transfer_model.save_weights(self.base_dir + '/transfer_model.h5')
        except Exception as e:
            print("Could not load model, {}".format(str(e))) 


    def load_weight(self):
        try:
            self.transfer_model.load_weights(self.base_dir + '/transfer_model.h5')
        except Exception as e:
            print("Save model failed, {}".format(str(e))) 


    def generate(self, content_imgs, style_imgs):
        return self.transfer_model.predict([content_imgs, style_imgs])


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