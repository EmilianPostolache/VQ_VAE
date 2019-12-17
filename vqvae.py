from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow_core.python.keras.utils import Progbar

from pixelcnn import PixelCNN, pixelcnn_loss
from utils import training_step, get_binarized_mnist


class VectorQuantization(Layer):
    def __init__(self, k, **kwargs):
        super(VectorQuantization, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        self.d = input_shape[-1]
        initializer = tf.keras.initializers.GlorotUniform()
        self.codebook = self.add_weight(name='codebook', shape=(self.k, self.d),
                                        initializer=initializer, trainable=True)

    def call(self, z_e, **kwargs):
        z_e_sg = tf.stop_gradient(z_e)
        cb = tf.reshape(self.codebook, (1, 1, 1, self.k, self.d))
        z_e_sg = z_e_sg[..., tf.newaxis, :]
        diff = z_e_sg - cb
        norm = tf.norm(diff, axis=-1)
        q_z = tf.argmin(norm, axis=-1)
        return q_z

    def sample(self, q_z):
        one_hot = tf.one_hot(q_z, self.k)
        one_hot = one_hot[..., tf.newaxis, :]
        cb = tf.transpose(self.codebook)
        cb = tf.reshape(cb, (1, 1, 1, self.d, self.k))
        z_q = tf.reduce_sum(cb * one_hot, axis=-1)
        return z_q


class Encoder(Model):
    def __init__(self, d, hidden_layers, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = []
        for n in hidden_layers:
            self.hidden_layers.append(Conv2D(n, 3, strides=(2, 2), padding='SAME', activation='relu'))
        self.output_layer = Conv2D(d, 3, strides=(1, 1), padding='SAME')

    def call(self, x, **kwargs):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        z_e = self.output_layer(x)
        return z_e


class Decoder(Model):
    def __init__(self, x_size, hidden_layers, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = []
        for n in hidden_layers:
            self.hidden_layers.append(Conv2DTranspose(n, 3, strides=(2, 2), padding='SAME', activation='relu'))
        self.output_layer = Conv2DTranspose(x_size, 3, strides=(1, 1), padding='SAME')

    def call(self, x, **kwargs):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        logits = self.output_layer(x)
        return logits


class VQVAE(Model):
    def __init__(self, d, k):
        super(VQVAE, self).__init__()
        self.d = d
        self.k = k
        self.encoder = Encoder(d, hidden_layers=[16, 32])
        self.vq = VectorQuantization(k)
        self.decoder = Decoder(1, hidden_layers=[32, 16])
        # PixelCNN dimension is latent dimension: 7
        self.pixelcnn = PixelCNN(7, 4, 16, d)

    def call(self, inputs, **kwargs):
        z_e = self.encoder(inputs)
        q_z = self.vq(z_e)
        z_q = self.vq.sample(q_z)
        z_q_sg = z_e + tf.stop_gradient(z_q - z_e)
        logits = self.decoder(z_q_sg)
        return logits, z_e, z_q, q_z

    def sample(self):
        q_z = self.pixelcnn.sample()
        z_q = self.vq.sample(q_z)
        logits = self.decoder(z_q)
        return tf.nn.softmax(logits)


@tf.function
def vqvae_loss(input, model, beta=0.25):
    logits, z_e, z_q, _ = model(input)

    print('z_e', z_e.shape)
    print('z_q', z_q.shape)
    # log p(x|z_q) reconstruction loss
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(input, logits)
    logp_x_z = tf.reduce_sum(cross_entropy, axis=[1, 2, 3])

    # vector quantization loss
    vq_loss = tf.norm(tf.stop_gradient(z_e) - z_q, axis=-1) ** 2
    vq_loss = tf.reduce_sum(vq_loss, axis=[1, 2])

    # # commitment loss
    commitment_loss = tf.norm(z_e - tf.stop_gradient(z_q), axis=-1) ** 2
    commitment_loss = tf.reduce_sum(commitment_loss, axis=[1, 2])
    return tf.reduce_mean(logp_x_z + vq_loss + beta*commitment_loss)


if __name__ == '__main__':

    IMAGE_SIZE = 28
    EPOCHS_VQVAE = 10
    EPOCHS_PIXELCNN = 10
    BATCH_SIZE = 1024
    BUFFER_SIZE = 60000
    D = 64
    K = 100
    BETA = 0.25

    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))
    SAVE_FILE = './checkpoints/vqvae_' + timestamp
    TENSORBOARD_DIR = './tensorboard/' + timestamp

    file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
    train_dataset, test_dataset, train_size, _ = get_binarized_mnist(BATCH_SIZE, BUFFER_SIZE)

    vqvae = VQVAE(D, K)
    adam = tf.keras.optimizers.Adam()
    loss_mean = tf.keras.metrics.Mean()

    for epoch in range(1, EPOCHS_VQVAE + 1):
        prog = Progbar(train_size / BATCH_SIZE)

        # train
        for step, input in enumerate(train_dataset):
            loss = training_step(input, vqvae, adam, vqvae_loss)
            prog.update(step, [('loss', loss)])

        # test
        for input in test_dataset:
            loss_mean(vqvae_loss(input, vqvae))
        print(f'\nVQVAE - Epoch {epoch} - Test NLL loss: {loss_mean.result().numpy()}')
        loss_mean.reset_states()

    # then you have to train pixelcnn
    for epoch in range(1, EPOCHS_PIXELCNN + 1):
        prog = Progbar(train_size / BATCH_SIZE)

        # train
        for step, input in enumerate(train_dataset):
            _, _, _, q_z = vqvae(input)
            q_z = q_z[..., tf.newaxis]
            loss = training_step(q_z, vqvae.pixelcnn, adam, pixelcnn_loss)
            prog.update(step, [('loss', loss)])

        # test
        for input in test_dataset:
            _, _, _, q_z = vqvae(input)
            q_z = q_z[..., tf.newaxis]
            loss_mean(pixelcnn_loss(q_z, vqvae.pixelcnn))
        print(f'\nPixelCNN - Epoch {epoch} - Test NLL loss: {loss_mean.result().numpy()}')
        loss_mean.reset_states()
