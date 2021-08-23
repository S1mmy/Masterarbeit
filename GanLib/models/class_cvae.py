# Basiert auf https://github.com/kn1cht/tensorflow_v2_cvae_sample/blob/master/sample-cvae-mnist.ipynb
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
from models.classy import CLASSY_SVHN
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from extra_keras_datasets import svhn

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

num_examples_to_generate = 100

model = CVAE(64, 10)

CLASSY_SVHN.load_weights('./classifier_checkpoint/best_cnn.h5')
CLASSY_SVHN.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True),
	  loss='categorical_crossentropy',
	  metrics=['accuracy'])
	  
(x_train, y_train), (x_test, y_test) = svhn.load_data(type='normal')

def preprocess_images(images):
  return tf.cast(images.reshape((images.shape[0], 32, 32, 3)) / 255., tf.float32)



x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)

train_size = x_train.shape[0] # = 60000
batch_size = 128
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



test_sample = []
for test_batch in test_dataset_xy.take(100):
  test_sample.append(test_batch[0][0:num_examples_to_generate, :,  :, :])
  test_sample.append(test_batch[1][0:num_examples_to_generate])



model.decoder.load_weights('./save/cvae_checkpoint.h5')

for i in range(0,100,2):
	mean, logvar = model.encode(test_sample[i],test_sample[i+1])
	z = model.reparameterize(mean, logvar)
	predictions = model.sample(z, test_sample[i+1])


	classes = (CLASSY_SVHN.predict(predictions).argmax(axis=1) + 1) % 10
	print(classes)
'''
fig = plt.figure(figsize=(10, 10))
for i in range(predictions.shape[0]):
	plt.subplot(10, 10, i + 1)
	plt.imshow(predictions[i, :, :, :])
	plt.title(classes[i])
	plt.axis('off')


fig.savefig("classify_cvae.png")
'''