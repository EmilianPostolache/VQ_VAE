import tensorflow as tf


def get_binarized_mnist(batch_size, buffer_size):
    # dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., tf.newaxis].astype('float32')
    x_test = x_test[..., tf.newaxis].astype('float32')

    # we have to perform quantization of the pixels
    x_train /= 255.
    x_test /= 255.

    x_train[x_train >= 0.5] = 1
    x_train[x_train < 0.5] = 0
    x_test[x_test >= 0.5] = 1
    x_test[x_test < 0.5] = 0

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)  # .shuffle(buffer_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)  # .shuffle(buffer_size)
    return train_dataset, test_dataset, x_train.shape[0], x_test.shape[0]


@tf.function
def training_step(x, model, optimizer, loss_function):
    with tf.GradientTape() as tape:
        loss = loss_function(x, model)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
