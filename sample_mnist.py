import tensorflow as tf
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

size = 8

plt.figure(figsize = (size,size))
gs1 = gridspec.GridSpec(size, size)
gs1.update(wspace=0.025, hspace=0.05) #

for i in range(8*8):  
  ax1 = plt.subplot(8,8,i+1)
  plt.imshow(x_train[i],cmap="gray")
  plt.axis('off')
  ax1.set_aspect('equal')

pyplot.tight_layout()

pyplot.show()