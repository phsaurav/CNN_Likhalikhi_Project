# region #*Importing Library
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
import random
import cv2
import imutils
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dense, MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# endregion

# region #*Importing Dataset
path = 'Dataset/Train/'
train_data = []
non_chars = ["#", "$", "&", "@"]

for i in os.listdir(path):
    if i in non_chars:
        continue
    count = 0
    sub_directory = os.path.join(path, i)
    for j in os.listdir(sub_directory):
        count += 1
        if count > 10000:
            break

        img = cv2.imread(os.path.join(sub_directory, j), 0)
        img = cv2.resize(img, (32, 32))
        train_data.append([img, i])

print(len(train_data))

val_path = 'Dataset/Validation/'
val_data = []

for i in os.listdir(val_path):
    if i in non_chars:
        continue
    count = 0
    sub_directory = os.path.join(val_path, i)
    for j in os.listdir(sub_directory):
        count += 1
        if count > 500:
            break
        img = cv2.imread(os.path.join(sub_directory, j), 0)
        img = cv2.resize(img, (32, 32))
        val_data.append([img, i])

print(len(val_data))
# endregion

# region #*Preprocessing
random.shuffle(train_data)
random.shuffle(val_data)

train_X = []
train_Y = []

for features, label in train_data:
    train_X.append(features)
    train_Y.append(label)

val_X = []
val_Y = []

for features, label in val_data:
    val_X.append(features)
    val_Y.append(label)

lb = LabelBinarizer()
train_Y = lb.fit_transform(train_Y)
val_Y = lb.fit_transform(val_Y)

train_X = np.array(train_X)/255.0
train_X = train_X.reshape(-1, 32, 32, 1)
train_Y = np.array(train_Y)

val_X = np.array(val_X)/255.0
val_X = val_X.reshape(-1, 32, 32, 1)
val_Y = np.array(val_Y)

print(train_X.shape, val_X.shape)
print(train_Y.shape, val_Y.shape)

# region #*Model Structrue
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(35, activation='softmax'))

model.summary()
# endregion

# region #*Compilation and Fitting
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# region #*Callbacks
filepath = 'weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

callbacks_list = [checkpoint, early_stop, log_csv]
# endregion

model.fit(train_X, train_Y, epochs=1, batch_size=32, validation_data=(val_X, val_Y), callbacks=callbacks_list, verbose=1)
#endregion 

# # region #*Saving Dataset
model.save('mnist_2.h5')
print("Saving the model as mnist_2.h5")
# # endregion
