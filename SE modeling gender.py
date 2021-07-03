#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten,Conv2D,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import collections
import cv2
import os
import hickle as hkl
import time


# read data from folder for train
datatrain_di="C:/SE modeling/gender/train/" # Directory of training data
CATEGORIES=['men','women']

for category in CATEGORIES:
    path=os.path.join(datatrain_di,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))

# assign resize value        
IMG_SIZE=100

# prepare training data
training_data=[]
def creat_training_data():
    for category in CATEGORIES:
        path=os.path.join(datatrain_di,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img))
            new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            training_data.append([new_array,class_num])
creat_training_data()

#Creat list for training data 
X_train=[]
Y_train=[]
for features,label in training_data:
    X_train.append(features)
    Y_train.append(label)
    
# reshape    
X_train=np.array(X_train).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y_train=np.array(Y_train)

# creat testin data
datatest_di="C:/SE modeling/gender/test/" # directory of testing data
test_data=[]
def creat_test_data():
    for category in CATEGORIES:
        path=os.path.join(datatest_di,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img))
            new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            test_data.append([new_array,class_num])
creat_test_data()

# creat list for training 
X_test=[]
Y_test=[]
for features,label in test_data:
    X_test.append(features)
    Y_test.append(label)
#reshape
X_test=np.array(X_test).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y_test=np.array(Y_test)

#modify data dimenation 
X_train = X_train.reshape(X_train.shape[0], 100, 100, 3)

#Normalizing data
X_train=X_train/255
X_test=X_test/255


from datetime import datetime
now1= datetime.now()

#Creat Model
model=Sequential()
model.add( Conv2D(32,(3,3),input_shape=(100,100,3)) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#Compile model
model.compile(loss="binary_crossentropy",
             optimizer='Adam',
             metrics=['accuracy'])
#fitting the model

history = model.fit(X_train,Y_train,batch_size=16,validation_split=0.2,epochs=20)


# In[7]:


now2=datetime.now()
d=now2-now1
d


# In[8]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[2]:


model.summary()


# In[3]:


train_acc = model.evaluate(X_train, Y_train, verbose=0)
train_acc


# In[4]:


#Pridect Men=0 Women=1 
Y_predict = model.predict_classes(X_test)
Y_predict


# In[3]:


#saving training data
data = {'X_train': X_train,'ytrain': Y_train}
hkl.dump(data,'gender.hkl')


# In[4]:


# Save the entire model as a SavedModel.
get_ipython().system('mkdir -p saved_model')
model.save('saved_model/my_model')


# In[56]:


#load model
load_model = tf.keras.models.load_model('saved_model/my_model')


# In[ ]:




