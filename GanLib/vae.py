import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
import time
import tensorflow_datasets as tfds
from models.cvae import CVAE, optimizer, log_normal_pdf, compute_loss, train_step

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def normalizeSVHN(x):
	image = tf.cast(x['image'], tf.float32)
	image = (image - 127.5) / 127.5
	return image

ds = tfds.load("svhn_cropped", with_info=True, data_dir='data/tensorflow_datasets', split='train', as_supervised=True, shuffle_files=True)

dst = tfds.load("svhn_cropped", with_info=True, data_dir='data/tensorflow_datasets', split='test', as_supervised=True, shuffle_files=True)
	
train_dataset_xy = tf.data.Dataset.zip(ds)
train_dataset_xy = train_dataset_xy.shuffle(60000).batch(batch_size)
test_dataset_xy = tf.data.Dataset.zip(dst)
test_dataset_xy = test_dataset_xy.shuffle(10000).batch(batch_size)

epochs = 100
latent_dim = 64
num_examples_to_generate = 16
label_size = 10

model = CVAE(latent_dim, label_size)


def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(*test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z, test_sample[1])
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, :])
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('vae/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
for test_batch in test_dataset_xy.take(1):
  test_sample = (test_batch[0][0:num_examples_to_generate, :,  :, :], test_batch[1][0:num_examples_to_generate])  


generate_and_save_images(model, 0, test_sample) 
 
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
  generate_and_save_images(model, epoch, test_sample)