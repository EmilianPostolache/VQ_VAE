# from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import Progbar

import numpy as np
from tensorflow.python.ops import array_ops
# from utils import training_step, get_binarized_mnist


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
                prob_cond = tf.nn.softmax(logits[0, i, j, :])
                current_sample[0, i, j, 0] = self.multinouli_sample(prob_cond)
                prog.update(i*self.x_size+j)
        return current_sample

    @staticmethod
    def multinouli_sample(dist):
        dist = np.array(dist)
        dist /= np.sum(dist)
        return np.random.choice(len(dist), p=dist)


@tf.function
def pixelcnn_loss(inputs, model):
    logits = model(inputs)
    inputs = tf.cast(inputs[:, :, :, 0], tf.int32)
    nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs, logits=logits)
    loss = tf.reduce_sum(nll, axis=[1, 2])
    return tf.reduce_mean(loss)


# if __name__ == '__main__':
#     IMAGE_SIZE = 28
#     FEATURE_MAPS = 16
#     OUTPUT_MAPS = 2
#     DEPTH = 4
#     EPOCHS = 10
#     BATCH_SIZE = 1024
#     BUFFER_SIZE = 60000
#
#     now = datetime.now()
#     timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))
#     SAVE_FILE = './checkpoints/pixelcnn_' + timestamp
#     TENSORBOARD_DIR = './tensorboard/' + timestamp
#
#     # file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
#     train_dataset, test_dataset, train_size, test_size = get_binarized_mnist(BATCH_SIZE, BUFFER_SIZE)
#
#     pixel_cnn = PixelCNN(IMAGE_SIZE, DEPTH, FEATURE_MAPS, OUTPUT_MAPS)
#     adam = tf.keras.optimizers.Adam()
#     loss_mean = tf.keras.metrics.Mean()
#
#     for epoch in range(1, EPOCHS + 1):
#         print(f'\nPixelCNN - Epoch {epoch}')
#
#         # train
#         prog = Progbar(train_size / BATCH_SIZE)
#         for step, input in enumerate(train_dataset):
#             loss = training_step(input, pixel_cnn, adam, pixelcnn_loss)
#             # logits = pixel_cnn(input)
#             # with file_writer.as_default():
#             #    tf.summary.image('input', input[0][tf.newaxis, ...], step=step+1)
#             #    tf.summary.histogram('input_hist', input[0, :, :, 0], step=step+1)
#             #    tf.summary.image('prob', tf.nn.sigmoid(logits)[0][tf.newaxis, ...], step=step+1)
#             #    tf.summary.histogram('prob_hist', tf.nn.sigmoid(logits)[0, :, :, 0], step=step+1)
#             loss_mean(loss)
#             prog.update(step, [('loss', loss)])
#         print(f'Train loss: {loss_mean.result().numpy()}')
#         loss_mean.reset_states()
#
#         # test
#         prog = Progbar(train_size / BATCH_SIZE)
#         for step, input in enumerate(test_dataset):
#             loss = pixelcnn_loss(input, pixel_cnn)
#             loss_mean(loss)
#             prog.update(step, [('loss', loss)])
#         print(f'Test loss: {loss_mean.result().numpy()}')
#         loss_mean.reset_states()
#
#         # sample = pixel_cnn.sample()
#         # with file_writer.as_default():
#         #     tf.summary.image('sample', sample, step=epoch)
#         #     tf.summary.histogram('sample_hist', sample, step=epoch)
