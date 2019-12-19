from datetime import datetime

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from vqvae import VQVAE, vqvae_loss
from pixelcnn import pixelcnn_loss
from utils import get_binarized_mnist  # , profile


@tf.function
def training_vqvae(x, model, optimizer, loss_function):
    with tf.GradientTape() as tape:
        loss = loss_function(x, model)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def training_pixelcnn(x, model, optimizer, loss_function):
    with tf.GradientTape() as tape:
        loss = loss_function(x, model)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(vqvae, file_writer):
    adam = tf.keras.optimizers.Adam()
    loss_mean = tf.keras.metrics.Mean()
    train_dataset, test_dataset, train_size, test_size = get_binarized_mnist(BATCH_SIZE, BUFFER_SIZE)

    for epoch in range(1, EPOCHS_VQVAE + 1):
        print(f'\nVQVAE - Epoch {epoch}')
        # train
        prog = Progbar(train_size / BATCH_SIZE)
        for step, input in enumerate(train_dataset):
            loss = training_vqvae(input, vqvae, adam, vqvae_loss)
            loss_mean(loss)
            prog.update(step, [('loss', loss)])
        print(f'\nTrain loss: {loss_mean.result().numpy()}')
        with file_writer.as_default():
            tf.summary.scalar('VQ-VAE Train NLL', loss_mean.result().numpy(), step=epoch)
        loss_mean.reset_states()

        # test
        prog = Progbar(test_size / BATCH_SIZE)
        for step, input in enumerate(test_dataset):
            loss = vqvae_loss(input, vqvae)
            loss_mean(loss)
            logits, _, _, _ = vqvae(input)
            with file_writer.as_default():
                tf.summary.image('Input', input, max_outputs=6, step=step)
                tf.summary.image('Output', tf.nn.softmax(logits[:, :, :, 0])
                                  [..., tf.newaxis], max_outputs=6, step=step)
            prog.update(step, [('loss', loss)])
        print(f'\nTest loss: {loss_mean.result().numpy()}')
        with file_writer.as_default():
            tf.summary.scalar('VQ-VAE Test NLL', loss_mean.result().numpy(), step=epoch)
        loss_mean.reset_states()

        # then you have to train pixelcnn
    adam = tf.keras.optimizers.Adam()
    for epoch in range(1, EPOCHS_PIXELCNN + 1):
        print(f'\nPixelCNN - Epoch {epoch}')

        # train
        prog = Progbar(train_size / BATCH_SIZE)
        for step, input in enumerate(train_dataset):
            _, _, _, q_z = vqvae(input)
            q_z = tf.cast(q_z[..., tf.newaxis], tf.float32)
            loss = training_pixelcnn(q_z, vqvae.pixelcnn, adam, pixelcnn_loss)
            prog.update(step, [('loss', loss)])
            loss_mean(loss)
        print(f'\nTrain loss: {loss_mean.result().numpy()}')
        with file_writer.as_default():
            tf.summary.scalar('PixelCNN Train NLL', loss_mean.result().numpy(), step=epoch)
        loss_mean.reset_states()

        # test
        prog = Progbar(test_size / BATCH_SIZE)
        for step, input in enumerate(test_dataset):
            _, _, _, q_z = vqvae(input)
            q_z = tf.cast(q_z[..., tf.newaxis], tf.float32)
            loss = pixelcnn_loss(q_z, vqvae.pixelcnn)
            prog.update(step, [('loss', loss)])
            loss_mean(loss)
        print(f'\nTest loss: {loss_mean.result().numpy()}')
        with file_writer.as_default():
            tf.summary.scalar('PixelCNN Test NLL', loss_mean.result().numpy(), step=epoch)
        loss_mean.reset_states()

def sample(vqvae, n, file_writer):
    for step in range(n):
      with file_writer.as_default():
        tf.summary.image('Sample', vqvae.sample(), step=step)


if __name__ == '__main__':

    IMAGE_SIZE = 28
    EPOCHS_VQVAE = 10
    EPOCHS_PIXELCNN = 10
    BATCH_SIZE = 1024
    BUFFER_SIZE = 60000
    D = 64
    K = 100
    BETA = 0.25
    N_SAMPLES = 100

    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))
    SAVE_FILE = './checkpoints/vqvae_' + timestamp
    TENSORBOARD_DIR = './tensorboard/' + timestamp

    file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    vqvae = VQVAE(D, K)
    train(vqvae, file_writer)
    sample(vqvae, N_SAMPLES, file_writer)
    # vqvae.sample()
    # profile(vqvae.pixelcnn, file_writer, TENSORBOARD_DIR)
