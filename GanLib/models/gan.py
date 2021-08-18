# Implementierung orientiert sich an https://github.com/marload/GANs-TensorFlow2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import time

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

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
		self.latent_dim, self.save_interval = 100, 50
		self.epochs, self.batch_size = EPOCHS, BATCH_SIZE

		self.generator, self.discriminator = Generator(data_size, channel), Discriminator()

		self.gen_optimizer, self.disc_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999), tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)

		if channel == 1:
			self.train_dataset = train_data.map(normalizeMNIST).shuffle(60000).batch(BATCH_SIZE)
			self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
			self.seed = tf.random.normal([64, self.latent_dim])
		else:
			self.train_dataset = train_data.map(normalizeSVHN).shuffle(60000).batch(BATCH_SIZE)
			self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
			self.seed = tf.random.uniform([64, self.latent_dim], minval=-1, maxval=1)
		
		self.result_path = RESULT_PATH
		
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
		
	@tf.function
	def train_step(self,images):
		noise = tf.random.normal([self.batch_size, self.latent_dim])

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

class Generator(tf.keras.Model):
	def __init__(self, data_size, channel):
		super(Generator, self).__init__()

		self.dense_1 = tf.keras.layers.Dense(64, activation='relu')
		self.dense_2 = tf.keras.layers.Dense(128, activation='relu')
		self.dense_3 = tf.keras.layers.Dense(256, activation='relu')
		self.dense_4 = tf.keras.layers.Dense(512, activation='relu')
		self.dense_5 = tf.keras.layers.Dense(1024, activation='relu')
		self.dense_6 = tf.keras.layers.Dense(2048, activation='relu')
		self.dense_7 = tf.keras.layers.Dense(data_size * data_size * channel, activation='tanh')

		self.reshape = tf.keras.layers.Reshape((data_size, data_size, channel))

	def call(self, inputs):
		x = self.dense_1(inputs)
		x = self.dense_2(x)
		x = self.dense_3(x)
		x = self.dense_4(x)
		x = self.dense_5(x)
		x = self.dense_6(x)
		x = self.dense_7(x)
		return self.reshape(x)


class Discriminator(tf.keras.Model):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.flatten = tf.keras.layers.Flatten()
		self.dense_1 = tf.keras.layers.Dense(2048, activation='relu')
		self.dense_2 = tf.keras.layers.Dense(1024, activation='relu')
		self.dense_3 = tf.keras.layers.Dense(512, activation='relu')
		self.dense_4 = tf.keras.layers.Dense(256, activation='relu')
		self.dense_5 = tf.keras.layers.Dense(128, activation='relu')
		self.dense_6 = tf.keras.layers.Dense(64, activation='relu')
		self.dense_7 = tf.keras.layers.Dense(1, activation="sigmoid")

	def call(self, inputs):
		x = self.flatten(inputs)
		x = self.dense_1(x)
		x = self.dense_2(x)
		x = self.dense_3(x)
		x = self.dense_4(x)
		x = self.dense_5(x)
		x = self.dense_6(x)
		return self.dense_7(x)