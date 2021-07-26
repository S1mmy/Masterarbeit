import tensorflow as tf
import math
import numpy as np

DATASET = "svhn"  # dataset to train on
LOAD_WEIGHT = True

# model parameter
SQUEEZE_FACTOR = 4
K_GLOW = 16 # The amount of blocks per layer
L_GLOW = 3 # The amount of layers
IMG_SIZE = 32  # better to be mult of SQUEEZE_FACTOR
CHANNEL_SIZE = 3
ACTIVATION = tf.nn.relu6
KERNEL_INITIALIZER_CLOSE_VALUE = lambda x=0: tf.random_normal_initializer(x, 1e-4)
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()

# training parameters
LEARNING_RATE = 5e-4
# LAMBDA_LIPSCHITZ = 1e-3
DROPOUT_N = 0.1
BATCH_SIZE = 32
SHUFFLE_SIZE = 20000
EPOCHS = 1000
CHECKPOINT_PATH = "./checkpoints/weights"
TENSORBOARD_LOGDIR = "./logs/GLOW"

# dataset parameters
ALPHA_BOUNDARY = 0.05

# general
TF_EPS = tf.keras.backend.epsilon()