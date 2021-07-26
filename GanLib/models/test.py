from extra_keras_datasets import svhn
import numpy


(x_train, y_train), (x_test, y_test) = svhn.load_data(type='normal')

y_train = (y_train)%10

unique, counts = numpy.unique(y_train, return_counts=True)

print(dict(zip(unique, counts)))