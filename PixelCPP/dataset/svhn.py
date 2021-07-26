import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import gin
from extra_keras_datasets import svhn

AUTOTUNE = tf.data.experimental.AUTOTUNE


@gin.configurable
def dataset(batch_size, shuffle_buffer_size=10000):    
    (train_images, train_y), (test_images, test_y) = svhn.load_data(type='normal')

    def _process_image(image):
        return tf.cast(image, tf.float32) * (2.0 / 255) - 1.0

    train = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .map(_process_image, num_parallel_calls=AUTOTUNE)
        .shuffle(shuffle_buffer_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .map(_process_image, num_parallel_calls=AUTOTUNE)
        .shuffle(shuffle_buffer_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train, test
