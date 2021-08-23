# Basiert auf https://github.com/kn1cht/tensorflow_v2_cvae_sample/blob/master/sample-cvae-mnist.ipynb
'''
Ein CVAE wird auf SVHN trainiert und samplet Bilder nach results/
Der Checkpoint wird nach save/ gespeichert.
'''
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

'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

'''
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


				
def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(*test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z, test_sample[1])
  fig = plt.figure(figsize=(8, 8))

  for i in range(predictions.shape[0]):
    plt.subplot(8, 8, i + 1)
    plt.imshow(predictions[i, :, :, :])
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch))
  
  
# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset_xy.take(1):
  test_sample = (test_batch[0][0:num_examples_to_generate, :,  :, :], test_batch[1][0:num_examples_to_generate])
  

generate_and_save_images(model, 0, test_sample)

model.decoder.save_weights('./save/cvae_checkpoint.h5')

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train in train_dataset_xy:
    train_step(model, train, optimizer)
  end_time = time.time()

  loss_tr = tf.keras.metrics.Mean()
  for train in train_dataset_xy:
    loss_tr(compute_loss(model, train))
  elbo_tr = -loss_tr.result()

  loss_tst = tf.keras.metrics.Mean()
  for test in test_dataset_xy:
    loss_tst(compute_loss(model, test))
  elbo_tst = -loss_tst.result()
  
  print(f'Epoch: {epoch}, Train set ELBO: {elbo_tr}, Test set ELBO: {elbo_tst}, time elapse for current epoch: {end_time - start_time}')
  if epoch%5 == 0:
    generate_and_save_images(model, epoch, test_sample)
  
  
#save weights
model.decoder.save_weights('./save/cvae_checkpoint.h5')