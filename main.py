import tensorflow as tf
from tensorflow import keras
from wasserstein_gan.wgan import wgan
tf.random.set_seed(42)

# Init Data Variables
TRAIN_BUF = 60000
BATCH_SIZE = 512
TEST_BUF = 10000
DIMS = (28,28,1)

# load dataset
(train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()

# split dataset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0

# batch datasets
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE))

# Wasserstein GAN
W_GAN = wgan(TRAIN_BUF, BATCH_SIZE, TEST_BUF, DIMS)
W_GAN.fit(train_dataset,test_dataset)