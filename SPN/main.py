import libspn_keras as spnk
from tensorflow import keras
import matplotlib.pyplot as plt

%matplotlib inline

# Model Data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


spnk.set_default_sum_op(spnk.SumOpGradBackprop())

spnk.set_default_accumulator_initializer(
    keras.initializers.TruncatedNormal(stddev=0.5, mean=1.0)
)

sum_product_network = keras.Sequential([
  spnk.layers.NormalizeStandardScore(input_shape=(28, 28, 1)),
  spnk.layers.NormalLeaf(
      num_components=16, 
      location_trainable=True,
      location_initializer=keras.initializers.TruncatedNormal(
          stddev=1.0, mean=0.0)
  ),
  # Non-overlapping products
  spnk.layers.Conv2DProduct(
      depthwise=True, 
      strides=[2, 2], 
      dilations=[1, 1], 
      kernel_size=[2, 2],
      padding='valid'
  ),
  spnk.layers.Local2DSum(num_sums=16),
  # Non-overlapping products
  spnk.layers.Conv2DProduct(
      depthwise=True, 
      strides=[2, 2], 
      dilations=[1, 1], 
      kernel_size=[2, 2],
      padding='valid'
  ),
  spnk.layers.Local2DSum(num_sums=32),
  # Overlapping products, starting at dilations [1, 1]
  spnk.layers.Conv2DProduct(
      depthwise=True, 
      strides=[1, 1], 
      dilations=[1, 1], 
      kernel_size=[2, 2],
      padding='full'
  ),
  spnk.layers.Local2DSum(num_sums=32),
  # Overlapping products, with dilations [2, 2] and full padding
  spnk.layers.Conv2DProduct(
      depthwise=True, 
      strides=[1, 1], 
      dilations=[2, 2], 
      kernel_size=[2, 2],
      padding='full'
  ),
  spnk.layers.Local2DSum(num_sums=64),
  # Overlapping products, with dilations [2, 2] and full padding
  spnk.layers.Conv2DProduct(
      depthwise=True, 
      strides=[1, 1], 
      dilations=[4, 4], 
      kernel_size=[2, 2],
      padding='full'
  ),
  spnk.layers.Local2DSum(num_sums=64),
  # Overlapping products, with dilations [2, 2] and 'final' padding to combine 
  # all scopes
  spnk.layers.Conv2DProduct(
      depthwise=True, 
      strides=[1, 1], 
      dilations=[8, 8], 
      kernel_size=[2, 2],
      padding='final'
  ),
  spnk.layers.SpatialToRegions(),
  # Class roots
  spnk.layers.DenseSum(num_sums=10),
  spnk.layers.RootSum(return_weighted_child_logits=True)
])



completion_model = define_spn(
    sum_op=spnk.SumOpGradBackprop(logspace_accumulators=False)
)
completion_model.set_weights(sum_product_network.get_weights())
completion_model.compile(metrics=[keras.metrics.MeanSquaredError()])

completion_model.evaluate([y_test], y_test, verbose=2)
completion_out = model.predict([y_test])
print(completion_out.shape)

