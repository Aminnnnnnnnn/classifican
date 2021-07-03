from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential
#from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import collections
import cv2
import os
datadir="train/"
CATEGORIES=['ped','bic']

for category in CATEGORIES:
    path=os.path.join(datadir,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
IMG_SIZE=100
training_data=[]
def creat_training_data():
    for category in CATEGORIES:
        path=os.path.join(datadir,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img))
            new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            training_data.append([new_array,class_num])
creat_training_data()
X_train=[]
Y_train=[]
for features,label in training_data:
    X_train.append(features)
    Y_train.append(label)
X_train=np.array(X_train).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y_train=np.array(Y_train)
X_train = X_train.reshape(X_train.shape[0], 100, 100, 3)
X_train=X_train/255
model=Sequential()
model.add( Conv2D(16,(3,3),input_shape=(100,100,3)) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))


model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss="binary_crossentropy",
             optimizer='Adam',
             metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=128,validation_split=0.2,epochs=20)

model.save('saved_model/my_model')

