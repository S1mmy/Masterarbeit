import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
import time
from models.cvae import CVAE, optimizer, log_normal_pdf, compute_loss, train_step

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from extra_keras_datasets import svhn

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


(x_train, y_train), (x_test, y_test) = svhn.load_data(type='normal')

def preprocess_images(images):
	return tf.cast(images.reshape((images.shape[0], 32, 32, 3)) / 255., tf.float32)



x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)

train_size = x_train.shape[0] # = 60000
batch_size = 100
test_size = x_test.shape[0] # = 10000
label_size = 10

train_dataset_x = tf.data.Dataset.from_tensor_slices(x_train)
test_dataset_x = tf.data.Dataset.from_tensor_slices(x_test)

train_dataset_y= tf.data.Dataset.from_tensor_slices(y_train)
test_dataset_y = tf.data.Dataset.from_tensor_slices(y_test)

train_dataset_xy = tf.data.Dataset.zip((train_dataset_x, train_dataset_y))
train_dataset_xy = train_dataset_xy.shuffle(train_size).batch(batch_size)
test_dataset_xy = tf.data.Dataset.zip((test_dataset_x, test_dataset_y))
test_dataset_xy = test_dataset_xy.shuffle(train_size).batch(batch_size)

epochs = 300
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 64
num_examples_to_generate = 64

model = CVAE(latent_dim, label_size)

model.decoder.load_weights('./save/cvae_checkpoint.h5')

def plot_latent_images(model, n, digit_size=32):
  """Plots n x n digit images decoded from the latent space."""

  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size*n
  image_height = image_width
  channel = 3
  image = np.zeros((image_height, image_width, 3))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z = np.array([[tf.constant(1/xi)]*64]).reshape(1, latent_dim)
      x_decoded = model.sample(z)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size, channel))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image)
  plt.axis('Off')
  plt.show()
  
plot_latent_images(model,20)
'''
def show_pics(x,pics):
	fig = plt.figure(figsize=(8, 4))
	for i in range(pics.shape[0]):
		plt.subplot(4, 8, i + 1)
		plt.imshow(pics[i, :, :, :])
		plt.axis('off')
	
	plt.savefig('./results/samples/%d.png' % (x))
	
def continuous_gen(x, model, fst_ind, snd_ind, label, w, h):
	continuous = []
	n = w * h
	for i in range(n):
		middle_z = tf.add(tf.scalar_mul(float(n-i)/n, z[fst_ind]), tf.scalar_mul(float(i)/n, z[snd_ind]))
		continuous.append(model.sample(tf.reshape(middle_z, [1, latent_dim]), [label]))

	fig = plt.figure(figsize=(w, h))
	for i in range(n):
		plt.subplot(h, w, i + 1)
		plt.imshow(tf.reshape(continuous[i], [32, 32, 3]))
		plt.axis('off')
	plt.savefig('./results/samples/%d.png' % (x))

mean, logvar = model.encode(x_test[0:32], y_test[0:32])
z = model.reparameterize(mean, logvar)
#show_pics(0,model.sample(z, y_test[0:32]))

#show_pics(1,model.sample(z, np.ones(32, dtype='uint8')*8))

continuous_gen(10, model, 25, 28, 5, 6, 4)
'''