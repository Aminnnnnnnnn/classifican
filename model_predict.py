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

CATEGORIES=['ped','bic']
predict_path = 'predict_image'
IMG_SIZE=100 # all of the trained images have been normalised to this size

model = load_model('saved_model/my_model')

model.compile(loss="binary_crossentropy",
             optimizer='Adam',
             metrics=['accuracy'])

# load some images to predict

#img = cv2.imread('predict_image/4888007.jpg')
##img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
#img = cv2.resize(img,(100,100))
#img = np.reshape(img,[1,IMG_SIZE,IMG_SIZE,3])


def predict_img(img_path):
    global CATEGORIES
    global model
    global IMG_SIZE

    # maybe check that the file exists first

    img = cv2.imread(img_path)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = np.reshape(img,[1,IMG_SIZE,IMG_SIZE,3])
    classes = model.predict_classes(img)

    class_index = classes[0][0]
    
    print("Input image: %s" % img_path)
    print("Image Index is %d" % class_index)
    print("Image Category is %s" % CATEGORIES[class_index])


print("\nPicture of a Bike:")
predict_img('predict_image/4888007.jpg')


print("\nPicture of a Pedestrian:")
predict_img('predict_image/pedestrian_example.jpg')

print("\nPicture of a Dog:")
predict_img('predict_image/dogs.jpg')

