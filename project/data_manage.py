from keras.layers.core import Dropout
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt

os.chdir('Dataset')

if os.path.isdir('test/#') is False:
    os.makedirs('test/#')
    os.makedirs('test/$')
    os.makedirs('test/&')
    os.makedirs('test/@')
    os.makedirs('test/0')
    os.makedirs('test/1')
    os.makedirs('test/2')
    os.makedirs('test/3')
    os.makedirs('test/4')
    os.makedirs('test/5')
    os.makedirs('test/6')
    os.makedirs('test/7')
    os.makedirs('test/8')
    os.makedirs('test/9')
    os.makedirs('test/A')
    os.makedirs('test/B')
    os.makedirs('test/C')
    os.makedirs('test/D')
    os.makedirs('test/E')
    os.makedirs('test/F')
    os.makedirs('test/G')
    os.makedirs('test/H')
    os.makedirs('test/I')
    os.makedirs('test/J')
    os.makedirs('test/K')
    os.makedirs('test/L')
    os.makedirs('test/M')
    os.makedirs('test/N')
    os.makedirs('test/P')
    os.makedirs('test/Q')
    os.makedirs('test/R')
    os.makedirs('test/S')
    os.makedirs('test/T')
    os.makedirs('test/U')
    os.makedirs('test/V')
    os.makedirs('test/W')
    os.makedirs('test/X')
    os.makedirs('test/Y')
    os.makedirs('test/Z')

    for c in random.sample(glob.glob('Train/#/*'), 100):
        shutil.copy(c, 'test/#')
    for c in random.sample(glob.glob('Train/$/*'), 100):
        shutil.copy(c, 'test/$')
    for c in random.sample(glob.glob('Train/&/*'), 100):
        shutil.copy(c, 'test/&')
    for c in random.sample(glob.glob('Train/@/*'), 100):
        shutil.copy(c, 'test/@')
    for c in random.sample(glob.glob('Train/0/*'), 100):
        shutil.copy(c, 'test/0')
    for c in random.sample(glob.glob('Train/1/*'), 100):
        shutil.copy(c, 'test/1')
    for c in random.sample(glob.glob('Train/2/*'), 100):
        shutil.copy(c, 'test/2')
    for c in random.sample(glob.glob('Train/3/*'), 100):
        shutil.copy(c, 'test/3')
    for c in random.sample(glob.glob('Train/4/*'), 100):
        shutil.copy(c, 'test/4')
    for c in random.sample(glob.glob('Train/5/*'), 100):
        shutil.copy(c, 'test/5')
    for c in random.sample(glob.glob('Train/6/*'), 100):
        shutil.copy(c, 'test/6')
    for c in random.sample(glob.glob('Train/7/*'), 100):
        shutil.copy(c, 'test/7')
    for c in random.sample(glob.glob('Train/8/*'), 100):
        shutil.copy(c, 'test/8')
    for c in random.sample(glob.glob('Train/9/*'), 100):
        shutil.copy(c, 'test/9')
    for c in random.sample(glob.glob('Train/A/*'), 100):
        shutil.copy(c, 'test/A')
    for c in random.sample(glob.glob('Train/B/*'), 100):
        shutil.copy(c, 'test/B')
    for c in random.sample(glob.glob('Train/C/*'), 100):
        shutil.copy(c, 'test/C')
    for c in random.sample(glob.glob('Train/D/*'), 100):
        shutil.copy(c, 'test/D')
    for c in random.sample(glob.glob('Train/E/*'), 100):
        shutil.copy(c, 'test/E')
    for c in random.sample(glob.glob('Train/F/*'), 100):
        shutil.copy(c, 'test/F')
    for c in random.sample(glob.glob('Train/G/*'), 100):
        shutil.copy(c, 'test/G')
    for c in random.sample(glob.glob('Train/H/*'), 100):
        shutil.copy(c, 'test/H')
    for c in random.sample(glob.glob('Train/I/*'), 100):
        shutil.copy(c, 'test/I')
    for c in random.sample(glob.glob('Train/J/*'), 100):
        shutil.copy(c, 'test/J')
    for c in random.sample(glob.glob('Train/K/*'), 100):
        shutil.copy(c, 'test/K')
    for c in random.sample(glob.glob('Train/L/*'), 100):
        shutil.copy(c, 'test/L')
    for c in random.sample(glob.glob('Train/M/*'), 100):
        shutil.copy(c, 'test/M')
    for c in random.sample(glob.glob('Train/N/*'), 100):
        shutil.copy(c, 'test/N')
    for c in random.sample(glob.glob('Train/P/*'), 100):
        shutil.copy(c, 'test/P')
    for c in random.sample(glob.glob('Train/Q/*'), 100):
        shutil.copy(c, 'test/Q')
    for c in random.sample(glob.glob('Train/R/*'), 100):
        shutil.copy(c, 'test/R')
    for c in random.sample(glob.glob('Train/S/*'), 100):
        shutil.copy(c, 'test/S')
    for c in random.sample(glob.glob('Train/T/*'), 100):
        shutil.copy(c, 'test/T')
    for c in random.sample(glob.glob('Train/U/*'), 100):
        shutil.copy(c, 'test/U')
    for c in random.sample(glob.glob('Train/V/*'), 100):
        shutil.copy(c, 'test/V')
    for c in random.sample(glob.glob('Train/W/*'), 100):
        shutil.copy(c, 'test/W')
    for c in random.sample(glob.glob('Train/X/*'), 100):
        shutil.copy(c, 'test/X')
    for c in random.sample(glob.glob('Train/Y/*'), 100):
        shutil.copy(c, 'test/Y')
    for c in random.sample(glob.glob('Train/Z/*'), 100):
        shutil.copy(c, 'test/Z')

os.chdir('../')
