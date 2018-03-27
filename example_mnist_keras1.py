from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np


np.random.seed(123)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print X_train.shape

#plotting the first sample of X_train
# from matplotlib import pyplot as plt
# plt.imshow(X_train[0])

# When using the Theano backend, you must explicitly declare a dimension for the depth of the input image.
# transform/reshape our dataset from having shape (n, width, height) to (n, width, height, depth).


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print X_train.shape

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# Convert 1-dimensional class arrays to 10-dimensional class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print y_train.shape

# build the model
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
# 32, (3, 3): number of conv filters, num of rows in each kernel, num of col in each kernel
# input shape = (width, height, depth) of each digit image
# step size of (1,1) by default
print model.input_shape, model.output_shape

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fit the model
model.fit(X_train, y_train,
          batch_size=32, epochs=10, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print score