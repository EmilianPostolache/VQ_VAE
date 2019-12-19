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
def run(model):
    return model(tf.zeros((1, 28, 28, 1)))


def profile(model, file_writer, profiler_outdir):
    tf.summary.trace_on(graph=True, profiler=True)
    run(model)
    with file_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=profiler_outdir)
