import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import gin
import logging

from tqdm import trange, tqdm

from models.PixelCNNPP import PixelCNNPP
from utils.losses import logistic_mixture_loss

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from extra_keras_datasets import svhn

import matplotlib.pyplot as plt

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model_cls=PixelCNNPP
optimizer_cls=tf.keras.optimizers.Adam
learning_rate=0.0002
learning_rate_decay=0.999995
batch_size=64
max_epoch=5000
chkpt_to_keep=5
images_to_log=16
log_images_every=50
debug=False

log_dir = "logdir/logs/"


inputs_shape = tf.TensorShape([None, 32, 32, 3])

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
	learning_rate, max_epoch, learning_rate_decay
)

model = model_cls(inputs_shape)
model.build(inputs_shape)

optimizer = optimizer_cls(learning_rate_schedule)

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, log_dir, chkpt_to_keep, 1)
print(manager.latest_checkpoint)
restore_status = checkpoint.restore(manager.latest_checkpoint)
print("Open Checkpoint!")
if manager.latest_checkpoint:
	print("Resuming from %s" % manager.latest_checkpoint)
	restore_status.assert_existing_objects_matched()

	
def sample_imgs(model,images_to_log):
	print("Sample...")
	fig = plt.figure(figsize=(10, 10))

	for i in range(10*10):
		plt.subplot(10, 10, i + 1)
		samples = model.sample(images_to_log)
		plt.imshow(tf.cast((samples + 1.0) * 127.5, tf.uint8))
		plt.axis('off')

	fig.savefig("sample.png")
	
sample_imgs(model,images_to_log)