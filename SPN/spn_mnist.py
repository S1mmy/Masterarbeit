# Basierend auf: https://github.com/pronobis/libspn-keras/blob/master/examples/notebooks/Sampling%20with%20conv%20SPNs.ipynb
import libspn_keras as spnk
from tensorflow import keras

spnk.set_default_accumulator_initializer(
    spnk.initializers.Dirichlet()
)

import numpy as np
import tensorflow_datasets as tfds
from libspn_keras.layers import NormalizeAxes
import tensorflow as tf


def take_first(a, b):
  return tf.reshape(tf.cast(a, tf.float32), (-1, 28, 28, 1))

normalize = spnk.layers.NormalizeStandardScore(
    input_shape=(28, 28, 1), axes=NormalizeAxes.GLOBAL, 
    normalization_epsilon=1e-3
)

mnist_images = tfds.load(name="mnist", batch_size=32, split="train", as_supervised=True).map(take_first)

normalize.adapt(mnist_images) 
mnist_normalized = mnist_images.map(normalize)
location_initializer = spnk.initializers.PoonDomingosMeanOfQuantileSplit(
    mnist_normalized
)

def build_spn(sum_op, return_logits, infer_no_evidence=False):
  spnk.set_default_sum_op(sum_op)
  return spnk.models.SequentialSumProductNetwork([
    normalize,
    spnk.layers.NormalLeaf(
        num_components=4, 
        location_trainable=True,
        location_initializer=location_initializer,
        scale_trainable=True
    ),
    spnk.layers.Conv2DProduct(
        depthwise=False, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.Local2DSum(num_sums=256),
    spnk.layers.Conv2DProduct(
        depthwise=True, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.Local2DSum(num_sums=512),
    # Pad to go from 7x7 to 8x8, so that we can apply 3 more Conv2DProducts
    tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1))),
    spnk.layers.Conv2DProduct(
        depthwise=True, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.Local2DSum(num_sums=512),
    spnk.layers.Conv2DProduct(
        depthwise=True, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.Local2DSum(num_sums=1024),
    spnk.layers.Conv2DProduct(
        depthwise=True, 
        strides=[2, 2], 
        dilations=[1, 1], 
        kernel_size=[2, 2],
        padding='valid'
    ),
    spnk.layers.LogDropout(rate=0.5),
    spnk.layers.DenseSum(num_sums=10),
    spnk.layers.RootSum(return_weighted_child_logits=return_logits)
  ], infer_no_evidence=infer_no_evidence, unsupervised=False)

sum_product_network = build_spn(spnk.SumOpEMBackprop(), return_logits=True)
sum_product_network.summary()

import tensorflow_datasets as tfds

batch_size = 128

mnist_train = (
    tfds.load(name="mnist", split="train", as_supervised=True)
    .shuffle(1024)
    .batch(batch_size)
)

mnist_test = (
    tfds.load(name="mnist", split="test", as_supervised=True)
    .batch(100)
)

optimizer = spnk.optimizers.OnlineExpectationMaximization(learning_rate=0.05, accumulate_batches=1)
metrics = []
loss = spnk.losses.NegativeLogJoint()

sum_product_network.compile(loss=loss, metrics=metrics, optimizer=optimizer)

import tensorflow as tf 

sum_product_network.fit(mnist_train, epochs=20, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", min_delta=0.1, patience=2, factor=0.5)])
sum_product_network.evaluate(mnist_test)


sum_product_network_sample = build_spn(spnk.SumOpSampleBackprop(), return_logits=False, infer_no_evidence=True)
sum_product_network_sample.set_weights(sum_product_network.get_weights())


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(12., 12.))
grid = ImageGrid(
    fig, 111,
    nrows_ncols=(10, 10),
    axes_pad=0.1,
)

sample = sum_product_network_sample.zero_evidence_inference(100)

print("Sampling done... Now ploting results")
for ax, im in zip(grid, sample):
    ax.imshow(np.squeeze(im), cmap="gray")
plt.show()