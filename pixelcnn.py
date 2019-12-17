# import sys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import Progbar
# from tensorflow.keras import regularizers
# from tensorflow.keras.layers import Layer
# from PIL import Image
from datetime import datetime
import numpy as np
from tensorflow.python.ops import array_ops


class MonoMaskedConv2D(Conv2D):
    def __init__(self, filters, kernel_size, mask_type='A', **kwargs):
        super(MonoMaskedConv2D, self).__init__(filters, kernel_size, **kwargs)
        if mask_type not in ['A', 'B']:
            raise ValueError('Only masks of type A and B are supported')
        self.mask_type = mask_type
        self.mask = None

    def build(self, input_shape):
        super(MonoMaskedConv2D, self).build(input_shape)
        center = self.kernel_size[0] // 2
        self.mask = np.ones(self.kernel.shape).astype('float32')
        self.mask[center + 1:, :, :, :] = 0.
        self.mask[center, center+1:, :, :] = 0.

        if self.mask_type == 'A':
            self.mask[center, center, :, :] = 0.

    def call(self, inputs, **kwargs):
        outputs = self._convolution_op(inputs, self.kernel * self.mask)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class ResidualBlock(Model):
    def __init__(self, h_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.h_size = h_size
        self.conv1 = tf.keras.layers.Conv2D(2*self.h_size, 1, padding='same', activation='relu')
        self.conv2 = MonoMaskedConv2D(self.h_size, 3, mask_type='B', padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(2*self.h_size, 1, padding='same', activation='relu')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x += inputs
        return x


class ResidualBlockSequence(Model):
    def __init__(self, n, h_size, **kwargs):
        super(ResidualBlockSequence, self).__init__(**kwargs)
        self.n = n
        self.h_size = h_size
        self.res_list = [ResidualBlock(h_size) for _ in range(n)]

    def call(self, x, **kwargs):
        for res in self.res_list:
            x = res(x)
        return x


class PixelCNN(Model):

    def __init__(self, x_size, n, h_size, d_size, **kwargs):
        super(PixelCNN, self).__init__(**kwargs)
        self.x_size = x_size
        self.h_size = h_size
        self.d_size = d_size
        self.n = n
        self.conv1 = MonoMaskedConv2D(2*h_size, 7, mask_type='A', padding='same', activation='relu')
        self.resnet = ResidualBlockSequence(n, h_size)
        self.conv2 = Conv2D(2*h_size, 1, padding='same', activation='relu')
        self.conv3 = Conv2D(d_size, 1, padding='same')  # , activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.resnet(x)
        x = self.conv2(x)
        logits = self.conv3(x)
        return logits

    def sample(self):
        current_sample = np.zeros((self.x_size, self.x_size))[np.newaxis, ..., np.newaxis].astype(np.float32)
        print('\nSampling...')
        prog = Progbar(self.x_size ** 2)
        for i in range(self.x_size):
            for j in range(self.x_size):
                logits = self.call(current_sample)
                prob_cond = tf.nn.sigmoid(logits[0, i, j, 0])
                current_sample[0, i, j, 0] = self.multinomial_sample([1. - prob_cond, prob_cond])
                prog.update(i*self.x_size+j)
        return current_sample

    @staticmethod
    def multinomial_sample(dist):
        dist = np.array(dist)
        dist /= np.sum(dist)
        return np.random.choice(len(dist), p=dist)


@tf.function
def compute_loss(inputs, model):
    logits = model(inputs)
    nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits)
    loss = tf.reduce_sum(nll, axis=[1, 2, 3])
    return tf.reduce_mean(loss)


@tf.function
def training_step(x, model, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(x, model)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


if __name__ == '__main__':
    IMAGE_SIZE = 28
    FEATURE_MAPS = 16
    OUTPUT_MAPS = 1
    DEPTH = 4
    EPOCHS = 10
    BATCH_SIZE = 1000
    BUFFER_SIZE = 60000

    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))
    SAVE_FILE = './checkpoints/pixelcnn_' + timestamp
    TENSORBOARD_DIR = './tensorboard/' + timestamp

    file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    # dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis].astype('float32')
    x_test = x_test[..., np.newaxis].astype('float32')

    # we have to perform quantization of the pixels

    x_train /= 255.
    x_test /= 255.

    x_train[x_train >= 0.5] = 1
    x_train[x_train < 0.5] = 0
    x_test[x_test >= 0.5] = 1
    x_test[x_test < 0.5] = 0

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)

    pixel_cnn = PixelCNN(IMAGE_SIZE, DEPTH, FEATURE_MAPS, OUTPUT_MAPS)
    adam = tf.keras.optimizers.Adam()
    loss_mean = tf.keras.metrics.Mean()

    for epoch in range(1, EPOCHS + 1):
        prog = Progbar(x_train.shape[0] / BATCH_SIZE)

        # train
        for step, input in enumerate(train_dataset):
            loss = training_step(input, pixel_cnn, adam)

            logits = pixel_cnn(input)
            with file_writer.as_default():
                tf.summary.image('input', input[0][tf.newaxis, ...], step=step+1)
                tf.summary.histogram('input_hist', input[0, :, :, 0], step=step+1)
                tf.summary.image('prob', tf.nn.sigmoid(logits)[0][tf.newaxis, ...], step=step+1)
                tf.summary.histogram('prob_hist', tf.nn.sigmoid(logits)[0, :, :, 0], step=step+1)

            prog.update(step, [('loss', loss)])

        # test
        for input in test_dataset:
            loss_mean(compute_loss(input, pixel_cnn))
        print(f'\nEpoch {epoch} - NLL loss: {loss_mean.result().numpy()}')
        loss_mean.reset_states()

        sample = pixel_cnn.sample()
        with file_writer.as_default():
            tf.summary.image('sample', sample, step=epoch)
            tf.summary.histogram('sample_hist', sample, step=epoch)
