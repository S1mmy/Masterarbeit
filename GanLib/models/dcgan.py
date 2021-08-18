# Implementierung orientiert sich an https://github.com/marload/GANs-TensorFlow2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras import layers
from models.classy import CLASSY_SVHN

def discriminator_loss(loss_object, real_output, fake_output):
	real_loss = loss_object(tf.ones_like(real_output), real_output)
	fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss


def generator_loss(loss_object, fake_output):
	return loss_object(tf.ones_like(fake_output), fake_output)

	
def normalizeMNIST(x):
	image = tf.cast(x['image'], tf.float32)
	image = (image / 127.5) - 1
	return image

def normalizeSVHN(x):
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

		self.gen_optimizer, self.disc_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.5, beta_2=0.999, amsgrad=False), tf.keras.optimizers.Adam(1e-4, beta_1=0.2, beta_2=0.999, epsilon=1e-7, amsgrad=False)

		if channel == 1:
			self.train_dataset = train_data.map(normalizeMNIST).shuffle(60000).batch(BATCH_SIZE)
			self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
			self.save_interval = 100
			self.seed  = tf.random.normal([64, self.latent_dim])
		else:
			self.train_dataset = train_data.map(normalizeSVHN).shuffle(60000).batch(BATCH_SIZE)
			self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
			self.seed  = tf.random.uniform([64, self.latent_dim], minval=-1, maxval=1)
			self.save_interval = 25
		
		if EPOCHS == 0:
			self.classify()
			return
		
		self.channel = channel
		self.result_path = RESULT_PATH
		
	def classify(self):
		print("Classify starting:")
		self.generator.load_weights('./save/dcgan_checkpoint.h5')
		#self.generator.compile(optimizer='adam', loss = 'binary_crossentropy')

		print("Generator loaded!")
		CLASSY_SVHN.load_weights('./classifier_checkpoint/best_cnn.h5')
		CLASSY_SVHN.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

		print("Weights Loaded!")
		
		noise = tf.random.uniform([100, self.latent_dim], minval=-1, maxval=1)
		gen_imgs = self.generator(noise)
		
		classes = (CLASSY_SVHN.predict(gen_imgs * .5 + .5).argmax(axis=1) + 1) % 10
		
		fig = plt.figure(figsize=(10, 10))

		for i in range(gen_imgs.shape[0]):
			plt.subplot(10, 10, i + 1)
			plt.imshow(gen_imgs[i, :, :, :] * .5 + .5)
			plt.title(classes[i])
			plt.axis('off')

		fig.savefig("classify_dcgan.png")
		

		
	def train(self):
		for epoch in range(self.epochs):
			total_gen_loss, total_disc_loss, start = 0, 0, time.time()

			for images in self.train_dataset:
				gen_loss, disc_loss = self.train_step(images)

				total_gen_loss += gen_loss
				total_disc_loss += disc_loss

			print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, total_gen_loss / self.batch_size, total_disc_loss / self.batch_size))
			if epoch % self.save_interval == 0:
				save_imgs(epoch, self.result_path, self.generator, self.seed)	
		
		save_imgs(self.epochs, self.result_path, self.generator, self.seed)
		self.generator.save_weights('./save/dcgan_checkpoint.h5')
		
	@tf.function
	def train_step(self,images):
		noise = None
		if self.channel == 1:
			noise = tf.random.normal([self.batch_size, self.latent_dim])
		else:
			noise = tf.random.uniform([self.batch_size, self.latent_dim], minval=-1, maxval=1)

		with tf.GradientTape(persistent=True) as tape:
			generated_images = self.generator(noise)

			real_output = self.discriminator(images)
			generated_output = self.discriminator(generated_images)

			gen_loss = generator_loss(self.cross_entropy, generated_output)
			disc_loss = discriminator_loss(self.cross_entropy, real_output, generated_output)

		grad_disc = tape.gradient(disc_loss, self.discriminator.trainable_variables)
		grad_gen = tape.gradient(gen_loss, self.generator.trainable_variables)

		self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))
		self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))

		return gen_loss, disc_loss		

def G(data_size, channel, training=True):

	model = tf.keras.Sequential()
	
	if channel == 3:
		model.add(layers.Dense(2*2*512, use_bias=False, input_shape=(100,)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		# model.add(layers.Dropout(0.2))
		model.add(layers.Reshape((2, 2, 512)))

		model.add(layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		
	else:
		model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0))
		# model.add(layers.Dropout(0.2))
		model.add(layers.Reshape((7, 7, 256)))
		model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU(alpha=0))

	model.add(layers.Conv2DTranspose(channel, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
	print(model.output_shape)
	assert model.output_shape == (None, data_size, data_size, channel)
	
	return model

def D(data_size, channel):
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(32, (5,5), strides=(2,2), padding='same', input_shape=[data_size, data_size, channel]))
	model.add(layers.LeakyReLU())
	
	model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same'))
	model.add(layers.LayerNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
	model.add(layers.LayerNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Flatten())
	model.add(layers.Dense(1, activation='sigmoid'))

	return model