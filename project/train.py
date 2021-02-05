#region #*Importing all necessary Library
from keras.layers.convolutional import SeparableConvolution1D
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, metrics, optimizers
import os
os.environ['TF_KERAS'] = '1'
#endregion 

#region #*Data import and splitting
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
#endregion 

#region #*Adding a new dimention for compatibility
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#endregion 
print(x_train.shape, y_train.shape)

#region #*Converting targets lablel in 10 catagorical answer
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#Todo: Here we have to insert one line

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

print("x_train shape: ", x_train.shape)
print(y_train.shape)
print(x_train.shape[0], " train sample")
print(x_test.shape[0], " test sample")
#endregion 

#region #*Creating Network Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adadelta', metrics=['accuracy'])           #The optimizer use is not sure

model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))
print("The model has successfully trained")

#endregion 

#region #*Saving Dataset
# model.save('mnist.h5')
# print("Saving the model as mnist.h5")
#endregion 



