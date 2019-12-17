import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import Model

from pixelcnn import PixelCNN


class VectorQuantization(Layer):
    def __init__(self, k, **kwargs):
        super(VectorQuantization, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        self.d = input_shape[-1]
        initializer = tf.keras.initializers.GlorotUniform()
        self.codebook = self.add_weight(name='codebook', shape=(self.k, self.d),
                                        initializer=initializer, trainable=True)

    def call(self, inputs, **kwargs):
        cb = tf.reshape(self.codebook, (1, 1, 1, self.k, self.d))
        inputs = inputs[..., tf.newaxis, :]
        diff = inputs - cb
        norm = tf.norm(diff, axis=-1)
        index = tf.argmin(norm, axis=-1)
        return index

    def sample(self, q_z):
        one_hot = tf.one_hot(q_z)
        one_hot = one_hot[..., tf.newaxis, :]
        cb = tf.transpose(self.codebook)
        cb = tf.reshape(cb, (1, 1, 1, self.d, self.k))
        sample = tf.reduce_sum(cb * one_hot, axis=-1)
        return sample


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
        self.output_layer = Conv2DTranspose(x_size, 3, strides=(1, 1), padding='SAME', activation='sigmoid')

    def call(self, x, **kwargs):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x


class VQVAE(Model):
    def __init__(self, d, k):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(d, hidden_layers=[16, 32])
        self.vq = VectorQuantization(k)
        self.decoder = Decoder(1, hidden_layers=[32, 16])
        self.pixelcnn = PixelCNN(28, 4, 16, 2)

    def call(self, inputs, **kwargs):
        z_e = self.encoder(inputs)
        q_z = self.vq(z_e)
        z_q = self.vq.sample(q_z)
        z_q = z_e + tf.stop_gradient(z_q - z_e)
        x = self.decoder(z_q)
        return x

    def sample(self):
        self.



@tf.function
def compute_loss(input, model):
    


if __name__ == '__main__':
    ...
