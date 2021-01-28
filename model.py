import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import utils
import tensorflow.keras.backend as K

from tensorflow.keras.layers import (
    Input, Activation, Layer,
    UpSampling2D, Concatenate,
    Add, Conv2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19
from ops import (
    WaveLetPooling, WaveLetUnPooling, TfReduceSum,
    WhiteningAndColoring, get_predict_function,
    gram_matrix)
from dataloader import build_input_pipe

VGG_LAYERS = [
    'block1_conv1', 'block1_conv2',
    'block2_conv1', 'block2_conv2',
    'block3_conv1', 'block3_conv2',
    'block3_conv3', 'block3_conv4',
    'block4_conv1',
]

class WCT2:
    def __init__(
        self,
        image_size=256,
        lr=1e-3,
        show_interval=25,
        gram_loss_weight=1.0,
        checkpoint_path="wtc2.h5"
    ):
        self.base_dir = base_dir
        self.image_size = image_size
        self.rst = rst
        self.lr = lr
        self.show_interval = show_interval
        self.img_shape = (self.image_size, self.image_size, 3)
        self.checkpoint_path = checkpoint_path

        img = Input(self.img_shape)
        self.wct = self.build_wct_model()
        # ======= Loss functions ======= #
        recontruct_img = self.wct(img)

        self.trainer = Model(inputs=[img], outputs=[recontruct_img], name="trainer")
        self.trainer.add_loss(gram_loss_weight * self.gram_loss(img, recontruct_img))
        self.trainer.compile(optimizer=Adam(self.lr), loss=["mse"])

        self.init_transfer_sequence()

    def gram_loss(self, img, gen_img):
        feat_gens = self.encoder(gen_img)
        feats = self.encoder(img)

        gram_gen = [gram_matrix(f) for f in feat_gens]
        gram_in = [gram_matrix(f) for f in feats]
        num_style_layers = len(gram_gen)
        loss_list = [
            K.mean(K.square(gram_gen[i] - gram_in[i]))
            for i in range(num_style_layers)
        ]

        gram_loss = TfReduceSum()(loss_list) / num_style_layers
        return gram_loss

    def conv_block(self, x, filters, kernel_size, activation='relu', name=""):
        x = Conv2D(
            filters, kernel_size=kernel_size, strides=1,
            padding='same', activation=activation, name=name)(x)

        return x

    def copy_layer(self, x, kernel_size, model, layer, name):
        """
        Need to copy layer for unique name
        """
        origin_layer = model.get_layer(layer)
        filters = origin_layer.output_shape[-1]
        return self.conv_block(x, filters, kernel_size,
                               name=origin_layer.name + name)

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

        vgg_output = [
            vgg_model.get_layer(f'block{i}_conv1'.get_output_at(0))
            for i in {1, 2, 3, 4}
        ]

        self.encoder = Model(inputs=vgg_model.inputs,
                             outputs=vgg_output,
                             name='encoder')

        # ======= Encoder ======= #
        id_ = 0
        x = self.copy_layer(img, kernel_size, vgg_model,
                            VGG_LAYERS[0], name='_encode')
        for layer in VGG_LAYERS[1:]:
            x = self.copy_layer(x, kernel_size, vgg_model, layer, name='_encode')
            if layer in {'block1_conv2', 'block2_conv2', 'block3_conv4'}:
                to_append = [x]
                x, lh, hl, hh= WaveLetPooling('wave_let_pooling_{}'.format(id_))(x)
                to_append += [lh, hl, hh]
                skips.append(to_append)
                id_ += 1

        # ======= Decoder ======= #
        skip_id = 2
        for layer in VGG_LAYERS[::-1][:-1]:
            filters = vgg_model.get_layer(layer).output_shape[-1]
            name = layer + "_decode"
            if layer in {'block4_conv1', 'block3_conv1', 'block2_conv1'}:
                x = self.conv_block(x, filters // 2, kernel_size, name=name)
                original, lh, hl, hh = skips[skip_id]
                x = WaveLetUnPooling('wave_let_unpooling_{}'.format(skip_id))([x, lh, hl, hh, original])
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

    def get_callbacks(self):
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=2, verbose=1,
                mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
            )
        ]

    def train(self, train_tfrec, val_tfrec, epochs=1, batch_size=4):
        # HARDCODE train_size
        train_size = 10000

        train_data = build_input_pipe(train_tfrec, batch_size, preprocess_method="vgg19")
        val_data =  build_input_pipe(val_tfrec, batch_size, preprocess_method="vgg19")
        steps_per_epoch = train_size // batch_size

        self.history = self.trainer.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.get_callbacks()
        )

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
            print("Save model failed, {}".format(str(e)))

    def load_weight(self):
        try:
            self.wct.load_weights(self.base_dir + '/wct2.h5')
        except Exception as e:
            print("Could not load model, {}".format(str(e))) 

    def transfer(self, content_img, style_img, alpha=1.0):
        # ===== Encode ===== #
        # step 1.
        content_feat, style_feat = self.en_1([content_img]), self.en_1([style_img])
        content_feat = WhiteningAndColoring(alpha)([content_feat, style_feat])
        # step 2.
        content_feat, c_skips_1 = self.pool_1([content_feat])
        style_feat, s_skips_1 = self.pool_1([style_feat])
        c_skips_1 = [WhiteningAndColoring(alpha)([c_skips_1[i], s_skips_1[i]]) for i in range(4)]
        content_feat = WhiteningAndColoring(alpha)([content_feat, style_feat])
        # step 3.
        content_feat, c_skips_2 = self.pool_2([content_feat])
        style_feat, s_skips_2 = self.pool_2([style_feat])
        c_skips_2 = [WhiteningAndColoring(alpha)([c_skips_2[i], s_skips_2[i]]) for i in range(4)]
        content_feat = WhiteningAndColoring(alpha)([content_feat, style_feat])
        # step 4.
        content_feat, c_skips_3 = self.pool_3([content_feat])
        style_feat, s_skips_3 = self.pool_3([style_feat])
        c_skips_3 = [WhiteningAndColoring(alpha)([c_skips_3[i], s_skips_3[i]]) for i in range(4)]
        content_feat = WhiteningAndColoring(alpha)([content_feat, style_feat])

        # ===== Decode ===== #
        # step 1.
        content_feat, style_feat = self.de_1([content_feat]), self.de_1([style_feat])
        content_feat = WhiteningAndColoring(alpha)([content_feat, style_feat])
        # step 2.
        content_feat = self.unpool_1([content_feat] + c_skips_3)
        style_feat = self.unpool_1([style_feat] + s_skips_3)
        content_feat = WhiteningAndColoring(alpha)([content_feat, style_feat])
        content_feat = self.de_2([content_feat])
        style_feat = self.de_2([style_feat])
        # step 3.
        content_feat = self.unpool_2([content_feat] + c_skips_2)
        style_feat = self.unpool_2([style_feat] + s_skips_2)
        content_feat = WhiteningAndColoring(alpha)([content_feat, style_feat])
        content_feat = self.de_3([content_feat])
        style_feat = self.de_3([style_feat])

        content_feat = self.unpool_3([content_feat] + c_skips_1)
        style_feat = self.unpool_3([style_feat] + s_skips_1)

        content_feat = WhiteningAndColoring(alpha)([content_feat, style_feat])


        content_feat = self.final([content_feat])

        return content_feat.numpy()

    def init_transfer_sequence(self):
        # ===== encoder layers ===== #
        self.en_1 = get_predict_function(self.wct, ['in_img', 'block1_conv1_encode'], name='en_1')

        self.pool_1 = get_predict_function(
            self.wct,
            ['block1_conv2_encode', 'wave_let_pooling_0', 'block2_conv1_encode'],
            name='pool_1')

        self.pool_2 = get_predict_function(
            self.wct,
            ['block2_conv2_encode', 'wave_let_pooling_1', 'block3_conv1_encode'],
            name='pool_2')

        self.pool_3 = get_predict_function(
            self.wct,
            [
                'block3_conv2_encode', 'block3_conv3_encode',
                'block3_conv4_encode',
                'wave_let_pooling_2', 'block4_conv1_encode'
            ],
            name='pool_3')

        # ===== decoder layers ===== #
        self.de_1 = get_predict_function(self.wct, ['block4_conv1_decode'], name='de_1')

        self.unpool_1 = get_predict_function(
            self.wct,
            [
                'wave_let_unpooling_2', 'block3_conv4_decode',
                'block3_conv3_decode', 'block3_conv2_decode'
            ],
            name='unpool_1')

        self.de_2 = get_predict_function(self.wct, ['block3_conv1_decode'], name='de_2')

        self.unpool_2 = get_predict_function(
            self.wct,
            ['wave_let_unpooling_1', 'block2_conv2_decode'],
            name='unpool_2')

        self.de_3 = get_predict_function(self.wct, ['block2_conv1_decode'], name='de_3')

        self.unpool_3 = get_predict_function(
            self.wct,
            ['wave_let_unpooling_0', 'block1_conv2_decode'],
            name='unpool_3')

        self.final = get_predict_function(self.wct, ['output'], name='final')


    def generate(self, content_imgs, style_imgs):
        return self.wct.predict([content_imgs, style_imgs])
