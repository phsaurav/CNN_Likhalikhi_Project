#region #*Importing all Library
from keras.engine.base_layer import _collect_input_shape
from keras.layers.core import Dropout
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout, Reshape
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
#endregion 

#region #*Dataset Importing
train_path = 'Dataset/Train'
# valid_path = 'Dataset/test'
valid_path = 'Dataset/Validation'

train_batches = ImageDataGenerator(rescale=1/255.0).flow_from_directory(directory=train_path, target_size=(32, 32), class_mode='categorical', batch_size=20, shuffle=True, color_mode='grayscale')
valid_batches = ImageDataGenerator(rescale=1/255.0).flow_from_directory(
    directory=valid_path, target_size=(32, 32), class_mode='categorical', batch_size=5, shuffle=True, color_mode='grayscale')

print(train_batches.class_indices)
imgs, labels = next(train_batches)
print(labels)
print(imgs.shape)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img,cmap=matplotlib.cm.gray)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)
print(labels)

print(valid_batches.class_indices)
imgs, labels = next(valid_batches)
print(labels)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img,cmap=matplotlib.cm.gray)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)
print(labels)
#endregion 

#region #*Model Structure
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3),activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(35, activation='softmax'))

model.summary()
#endregion 

#region #*Compiling and Fitting Model
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.001),loss='categorical_crossentropy', metrics=['accuracy'])

#region #*Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
filepath = 'weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
log_csv = CSVLogger('my_logs_up_2_Adam.csv', separator=',', append=False)

callbacks_list = [checkpoint, early_stop, log_csv]
#endregion 

model.fit(x=train_batches, epochs=25, validation_data=valid_batches, callbacks=callbacks_list, verbose=1)
#endregion 

#region #*Saving Dataset
model.save('Model_up_2_Adam_25.h5')
print("Saving the model as Model_up_2.h5")
#endregion
