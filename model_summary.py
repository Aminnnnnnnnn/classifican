from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential,load_model
#from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import collections
import cv2
import os

#model = Sequential()
model = load_model('saved_model/my_model')
model.summary()

