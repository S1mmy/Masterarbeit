import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras import layers
import math
from models.classy import CLASSY_SVHN
from functools import partial

GP_WEIGHT = 10.0

'''
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
'''

def get_loss_fn():
	def d_loss_fn(real_logits, fake_logits):
		return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

	def g_loss_fn(fake_logits):
		return -tf.reduce_mean(fake_logits)

	return d_loss_fn, g_loss_fn

def normalize(x):
	image = tf.cast(x['image'], tf.float32)
	image = (image - 127.5) / 127.5
	return image

def save_imgs(epoch, result_path, generator, noise):
	gen_imgs = generator(noise)

	fig = plt.figure(figsize=(8, 8))

	for i in range(gen_imgs.shape[0]):
		plt.subplot(8, 8, i + 1)
		if "mnist" in result_path:
			plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5)
		else:
			plt.imshow(gen_imgs[i, :, :, :] * .5 + .5)
		plt.axis('off')

	fig.savefig("%s/%d.png" % (result_path,epoch))

class GAN:
	def __init__(self, train_data, EPOCHS, BATCH_SIZE, RESULT_PATH, data_size, channel):
		self.latent_dim = 100
		self.epochs, self.batch_size = EPOCHS, BATCH_SIZE

		self.generator, self.discriminator = G(data_size, channel), D(data_size, channel)

		self.gen_optimizer, self.disc_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.5, beta_2=0.999, epsilon=1e-7, amsgrad=False), tf.keras.optimizers.Adam(1e-4, beta_1=0.2, beta_2=0.999, epsilon=1e-7, amsgrad=False)
		
		self.train_dataset = train_data.map(normalize).shuffle(60000).batch(BATCH_SIZE)
		
		self.save_interval = 25
		
		self.seed  = tf.random.uniform([64, self.latent_dim], minval=-1, maxval=1)			
		
		self.channel = channel
		self.result_path = RESULT_PATH
		
		if EPOCHS == 0:
			self.classify()
			return
		
		self.d_loss_fn, self.g_loss_fn = get_loss_fn()
		
		
	def classify(self):
		print("Classify starting:")
		self.generator.load_weights('./save/dragan_checkpoint.h5')
		self.generator.compile(optimizer='adam', loss = 'binary_crossentropy')

		print("Generator loaded!")
		CLASSY_SVHN.load_weights('./classifier_checkpoint/best_cnn.h5')
		CLASSY_SVHN.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

		print("Weights Loaded!")
		for i in range(100):
			noise = tf.random.uniform([100, self.latent_dim], minval=-1, maxval=1)
			gen_imgs = self.generator(noise)
			
			print(CLASSY_SVHN.predict(gen_imgs * .5 + .5).argmax(axis=1))
		'''
		fig = plt.figure(figsize=(8, 8))

		for i in range(gen_imgs.shape[0]):
			plt.subplot(8, 8, i + 1)
			plt.imshow(gen_imgs[i, :, :, :] * .5 + .5)
			plt.axis('off')

		fig.savefig("classify_dcgan.png")
		'''
		
	def gradient_penalty(self, generator, real_images):
		real_images = tf.cast(real_images, tf.float32)
		def _interpolate(a):
			beta = tf.random.uniform(tf.shape(a), 0., 1.)
			b = a + 0.5 * tf.math.reduce_std(a) * beta
			shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
			alpha = tf.random.uniform(shape, 0., 1.)
			inter = a + alpha * (b - a)
			inter.set_shape(a.shape)
			return inter
		
		x = _interpolate(real_images)
		with tf.GradientTape() as tape:
			tape.watch(x)
			predictions = generator(x)
		grad = tape.gradient(predictions, x)
		slopes = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
		return tf.reduce_mean((slopes - 1.) ** 2)

	@tf.function
	def train_step(self,real_images):
		z = tf.random.uniform([self.latent_dim, self.batch_size],minval=-1, maxval=1)
		with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
			fake_images = self.generator(z)

			fake_logits = self.discriminator(fake_images)
			real_logits = self.discriminator(real_images)

			d_loss = self.d_loss_fn(real_logits, fake_logits)
			g_loss = self.g_loss_fn(fake_logits)

			gp = self.gradient_penalty(partial(self.discriminator), real_images)
			d_loss += gp * GP_WEIGHT

		d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
		g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)

		self.disc_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
		self.gen_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

		return g_loss, d_loss
	
	def train(self):
		for epoch in range(self.epochs):
			total_gen_loss, total_disc_loss, start = 0, 0, time.time()

			for images in self.train_dataset:
				gen_loss, disc_loss = self.train_step(images)
				
				if math.isnan(gen_loss):
					print("Nan Value Detected!")
					return

				total_gen_loss += gen_loss
				total_disc_loss += disc_loss

			print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, total_gen_loss / self.batch_size, total_disc_loss / self.batch_size))
			if epoch % self.save_interval == 0:
				save_imgs(epoch, self.result_path, self.generator, self.seed)	
		
		save_imgs(self.epochs, self.result_path, self.generator, self.seed)
		self.generator.save_weights('./save/dragan_checkpoint.h5')
		

def G(data_size, channel):

	model = tf.keras.Sequential()
	
	if channel == 3:
		model.add(layers.Dense(2*2*512, use_bias=False, input_shape=(100,)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		# model.add(layers.Dropout(0.2))
		model.add(layers.Reshape((2, 2, 512)))

		model.add(layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		
		model.add(layers.Conv2DTranspose(32, 5, strides=2, padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		
	else:
		model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		# model.add(layers.Dropout(0.2))
		model.add(layers.Reshape((7, 7, 256)))
		model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(channel, 5, strides=2, padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, data_size, data_size, channel)
	
	return model

def D(data_size, channel):
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(32, 5, strides=2, padding='same', input_shape=[data_size, data_size, channel]))
	model.add(layers.LeakyReLU())
	
	model.add(layers.Conv2D(64, 5, strides=2, padding='same'))
	model.add(layers.LayerNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2D(128, 5, strides=2, padding='same'))
	model.add(layers.LayerNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Flatten())
	model.add(layers.Dense(1))

	return model