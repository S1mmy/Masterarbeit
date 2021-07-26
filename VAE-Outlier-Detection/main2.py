import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Layer, Reshape, InputLayer
from tqdm import tqdm

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from extra_keras_datasets import svhn

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from alibi_detect.models.tensorflow.losses import elbo
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.fetching import fetch_detector
from alibi_detect.utils.perturbation import apply_mask
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

mask = (y_train <= 4)

X_train = X_train[mask]
y_train = y_train[mask]

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


filepath = "cvae_model_save/OutlierVAE"

latent_dim = 1024

load_outlier_detector = False

if load_outlier_detector:  # load pretrained outlier detector
	detector_type = 'outlier'
	dataset = 'svhn'
	detector_name = 'OutlierVAE'
	od = fetch_detector(filepath, detector_type, dataset, detector_name)
	#filepath = os.path.join(filepath, detector_name)
else:
	encoder_net = tf.keras.Sequential(
	  [
		  InputLayer(input_shape=(28, 28, 1)),
		  Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu),
		  Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
		  Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu)
	  ])

	decoder_net = tf.keras.Sequential(
	  [
		  InputLayer(input_shape=(latent_dim,)),
		  Dense(7*7*256),
		  Reshape(target_shape=(7, 7, 256)),
		  Conv2DTranspose(128, 4, strides=2, padding='same', activation=tf.nn.relu),
		  Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')
	  ])

	# initialize outlier detector
	od = OutlierVAE(threshold=.015,  # threshold for outlier score
					score_type='mse',  # use MSE of reconstruction error for outlier detection
					encoder_net=encoder_net,  # can also pass VAE model instead
					decoder_net=decoder_net,  # of separate encoder and decoder
					latent_dim=latent_dim,
					samples=2)
	# train
	od.fit(X_train,
		   loss_fn=elbo,
		   cov_elbo=dict(sim=.05),
		   epochs=200,
		   verbose=False)

	# save the trained outlier detector
	save_detector(od, filepath)

'''
idx = 8
X = X_train[idx].reshape(1, 32, 32, 3)
X_recon = od.vae(X)

plt.imshow(X.reshape(32, 32, 3))
plt.axis('off')
plt.savefig("original.png")

plt.imshow(X_recon.numpy().reshape(32, 32, 3))
plt.axis('off')
plt.savefig("reconstructed.png")

'''

old = X_train[0:2500]
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

mask = y_train == 9
X_train = X_train[mask]
y_train = y_train[mask]

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

X = np.vstack((old,X_train[0:1000]))

print(X.shape)

od_preds = od.predict(X,
                      outlier_type='instance',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)
print(list(od_preds['data'].keys()))

target = np.zeros(X.shape[0],).astype(int)  # all normal CIFAR10 training instances
labels = ['normal', 'outlier']
plot_instance_score(0, od_preds, target, labels, od.threshold)

X_recon = od.vae(X).numpy()
plot_feature_outlier_image(0,od_preds,
                           X,
                           X_recon=X_recon,
                           instance_ids=[8, 60, 100, 330],  # pass a list with indices of instances to display
                           max_instances=5,  # max nb of instances to display
                           outliers_only=False)  # only show outlier predictions
						   
od_preds = od.predict(X,
                      outlier_type='feature',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)
print(list(od_preds['data'].keys()))

target = np.zeros(X.shape[0],).astype(int)  # all normal CIFAR10 training instances
labels = ['normal', 'outlier']
plot_instance_score(1, od_preds, target, labels, od.threshold)

X_recon = od.vae(X).numpy()
plot_feature_outlier_image(1,od_preds,
                           X,
                           X_recon=X_recon,
                           instance_ids=[8, 60, 100, 330],  # pass a list with indices of instances to display
                           max_instances=5,  # max nb of instances to display
                           outliers_only=False)  # only show outlier predictions