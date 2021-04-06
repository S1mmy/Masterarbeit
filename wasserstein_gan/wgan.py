#Wasserstein GAN https://github.com/YBen1/Wasserstein-GAN-Tensorflow-2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from tqdm.autonotebook import tqdm
np.random.seed(42)
tf.random.set_seed(42)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class wgan:
	gen_learning_rate, disc_learning_rate = 0.0001, 0.0002
	number_of_disc_layers = 6

	def __init__(self, TRAIN_BUF, BATCH_SIZE, TEST_BUF, DIMS):
		self.DIMS = DIMS
		self.generator_optimizer = tf.keras.optimizers.RMSprop(self.gen_learning_rate)
		self.discriminator_optimizer = tf.keras.optimizers.RMSprop(self.disc_learning_rate)
		
		self.N_TRAIN_BATCHES = int(TRAIN_BUF/BATCH_SIZE)
		self.N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)
		
		self.generator = self.get_generator()
		self.discriminator = self.get_discriminator()

	#Generator Model
	def get_generator(self):
		generator = tf.keras.models.Sequential([
			tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
			tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
			tf.keras.layers.Conv2DTranspose(
				filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
			),
			tf.keras.layers.Conv2DTranspose(
				filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
			),
			tf.keras.layers.Conv2DTranspose(
				filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
			)
		])
		return generator
	
	#Discriminator Model
	def get_discriminator(self):
		discriminator = tf.keras.models.Sequential([
			tf.keras.layers.InputLayer(input_shape=self.DIMS),
			tf.keras.layers.Conv2D(
				filters=32, kernel_size=3, strides=(2, 2), activation="relu"
			),
			tf.keras.layers.Conv2D(
				filters=64, kernel_size=3, strides=(2, 2), activation="relu"
			),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(units=1, activation="sigmoid"),
		])
		return discriminator
		
	@tf.function
	def compute_loss(self,train_x):
		x  = tf.random.normal([train_x.shape[0], 1, 1, 64])

		real_output = self.discriminator(train_x)
		fake_output = self.discriminator(self.generator(x))
		disc_loss = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
		gen_loss = -tf.reduce_mean(fake_output)

		return disc_loss, gen_loss
		
	@tf.function
	def train_step(self,train_x,n_steps=4):
		x = tf.random.normal([train_x.shape[0], 1, 1, 64])
		for i in range(n_steps):
			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				real_output = self.discriminator(train_x)
				fake_output = self.discriminator(self.generator(x))

				disc_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

				gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
				self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
				
				t=0
				for t in range(self.number_of_disc_layers):
					y = tf.clip_by_value(self.discriminator.trainable_weights[t],clip_value_min=-0.05,clip_value_max=0.05,name=None)
					self.discriminator.trainable_weights[t].assign(y)

				if i == (n_steps-1) :
					fake_training_data = self.generator(x)
					fake_output = self.discriminator(fake_training_data)
					gen_loss = -tf.reduce_mean(fake_output)
					gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
					self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
			
	# exampled data for plotting results
	def plot_reconstruction(self,epoch,train_x, nex=8, zm=2):
		samples = self.generator(tf.random.normal([train_x.shape[0], 1, 1, 64]))
		fig, axs = plt.subplots(ncols=nex, nrows=1, figsize=(zm * nex, zm))
		for axi in range(nex):
			axs[axi].matshow(samples.numpy()[axi].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1)
			axs[axi].axis('off')
		plt.savefig('WGAN_'+str(epoch)+'.png')
 
	def final_export(self,losses):
		fig = plt.figure(figsize=(10, 6))
		plt.plot(losses.disc_loss.values) 
		plt.ylabel("Loss", fontsize=14, rotation=90)
		plt.xlabel("Iterations", fontsize=14)
		plt.legend(['WGAN training loss'], prop={'size': 14}, loc='upper right');
		plt.grid(True, which="both")
		plt.savefig('FINAL_OUTPUT_WGAN.png')
		
	def fit(self,train_dataset,test_dataset,n_epochs=150):
		losses = pd.DataFrame(columns = ['disc_loss', 'gen_loss'])
		start = time.time()
		for epoch in range(n_epochs):
			# train
			for batch, train_x in tqdm(zip(range(self.N_TRAIN_BATCHES), train_dataset), total=self.N_TRAIN_BATCHES):
				self.train_step(train_x)

			loss = []
			for batch, test_x in tqdm(
				zip(range(self.N_TEST_BATCHES), test_dataset), total=self.N_TEST_BATCHES
			):
				loss.append(self.compute_loss(train_x))
			losses.loc[len(losses)] = np.mean(loss, axis=0)

			print(
				"Epoch: {} | disc_loss: {} | gen_loss: {}".format(
					epoch, losses.disc_loss.values[-1], losses.gen_loss.values[-1]
				)
			)
			self.plot_reconstruction(epoch,train_x)

		time_to_train_gan = time.time()-start
		tf.print('Time for the training is {} sec,'.format( time.time()-start))
		
		self.final_export(losses)
