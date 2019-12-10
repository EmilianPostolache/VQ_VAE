import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import Progbar

import numpy as np


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

    def call(self, inputs):
        x = tf.nn.conv2d(inputs, self.kernel * self.mask, strides=self.strides, padding=str.upper(self.padding))
        if self.use_bias:
            x = x + self.bias
        return x


class ResidualBlock(Model):
    def __init__(self, h_size):
        super(ResidualBlock, self).__init__()
        self.h_size = h_size
        self.relu1 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv2D(2*self.h_size, 1, padding='SAME')
        self.relu2 = tf.keras.layers.ReLU()
        self.conv2 = MonoMaskedConv2D(self.h_size, 3, mask_type='B', padding='SAME')
        self.relu3 = tf.keras.layers.ReLU()
        self.conv3 = tf.keras.layers.Conv2D(2*self.h_size, 1, padding='SAME')

    def call(self, inputs, **kwargs):
        x = self.relu1(inputs)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.relu3(x)
        x = self.conv3(x)
        x += inputs
        return x


class PixelCNN(Model):

    def __init__(self, x_size, h_size, d_size):
        super(PixelCNN, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.d_size = d_size
        self.conv1 = MonoMaskedConv2D(2*h_size, x_size, mask_type='A', padding='SAME')
        self.resnet1 = ResidualBlock(h_size)
        self.resnet2 = ResidualBlock(h_size)
        self.resnet3 = ResidualBlock(h_size)
        self.resnet4 = ResidualBlock(h_size)
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = Conv2D(2*h_size, 1, padding='SAME')
        self.relu2 = tf.keras.layers.ReLU()
        self.conv3 = Conv2D(d_size, 1, padding='SAME')
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.resnet3(x)
        x = self.resnet4(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.softmax(x)
        return x

    def sample(self):
        current_sample = np.zeros((self.x_size, self.x_size))[np.newaxis, ..., np.newaxis].astype('float32')
        for i in range(self.x_size):
            for j in range(self.x_size):
                prob_cond = self.call(current_sample)
                current_sample[..., i, j, ...] = self.multinomial_sample(prob_cond[0, i, j, :])
        return current_sample

    def multinomial_sample(self, dist):
        return np.array([np.random.choice(self.d_size, p=d) for d in dist])


@tf.function
def compute_loss(inputs, model):
    p_conditional = model(inputs)
    # giacche vogliamo calcolare il likelihood dobbiamo selezionare per ogni distribuzione p(x_i| x_i-1, ... x_0)
    # il valore relativo al dato di input, dunque usiamo tf.gather_nd (aumentando di una dimensione l'input, per
    # matchare le dimensioni di batching, non solo la prima ma anche x_dim * x_dim
    p_conditional = tf.gather_nd(p_conditional, tf.cast(inputs, tf.int32), batch_dims=3)
    log_likelihood = - tf.reduce_sum(tf.math.log(p_conditional), axis=[1, 2])
    return tf.reduce_mean(log_likelihood)


@tf.function
def training_step(x, model, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(x, model)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


if __name__ == '__main__':

    BATCH_SIZE = 1000
    BUFFER_SIZE = 60000
    EPOCHS = 1
    IMAGE_SIZE = 32
    FEATURE_MAPS = 16
    OUTPUT_MAPS = 2

    from datetime import datetime
    now = datetime.now()
    timestamp = str(now.strftime("%Y-%m-%d_%H-%M-%S"))
    SAVE_FILE = './checkpoints/pixelcnn_' + timestamp

    # dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis].astype('float32')
    x_test = x_test[..., np.newaxis].astype('float32')

    x_train /= 255.
    x_test /= 255.

    x_train[x_train >= 0.5] = 1
    x_train[x_train < 0.5] = 0
    x_test[x_test >= 0.5] = 1
    x_test[x_test < 0.5] = 0

    # we have to perform quantization of the pixels

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    pixel_cnn = PixelCNN(IMAGE_SIZE, FEATURE_MAPS, OUTPUT_MAPS)
    adam = tf.keras.optimizers.Adam()

    loss_mean = tf.keras.metrics.Mean()

    for epoch in range(EPOCHS):
        prog = Progbar(x_train.shape[0] / BATCH_SIZE)

        # train
        for step, input in enumerate(train_dataset):
            loss = training_step(input, pixel_cnn, adam)
            prog.update(step, [('loss', loss)])

        # test
        for input in test_dataset:
            loss_mean(compute_loss(input, pixel_cnn))

        print(f'\nEpoch {epoch} - NLL loss: {loss_mean.result().numpy()}')
        loss_mean.reset_states()

    pixel_cnn.save_weights(SAVE_FILE)
    print('Model weights have been successfully saved in ' + SAVE_FILE)
