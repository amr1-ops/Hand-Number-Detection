# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:01:53 2022

@author: Ahmed Amr
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers,models

DATADIR = "Dataset"

CATEGORIES = ["0","1","2","3","4","5"]

data = []
def create_data():
    for category in CATEGORIES:  # do

        path = os.path.join(DATADIR,category)  # create path 
        class_num = CATEGORIES.index(category)  # get the classification

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                img_array = np.array(img_array)
                data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
       
create_data()


y = []
features_data = []


for feature,label in data:
    y.append(label)
    features_data.append(feature)  



features_data = np.array(features_data)

Xtrain, Xtest, ytrain, ytest = train_test_split(features_data,
                                                    y,
                                                    test_size=.25,
                                                    random_state=1234123)


Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
ytrain = np.array(ytrain)
ytest = np.array(ytest)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),activation='relu', input_shape=(300,300,3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#--------------------------------------------

model.add(layers.Dropout(0.1))

model.add(layers.Flatten())  # this converts our 3D feature(width,height,channel) maps to 1D feature vectors

model.add(layers.Dense(64,activation='relu'))#feature selection

model.add(layers.Dense(6))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])#combination between two GD methodologies "adam"


history = model.fit(Xtrain, ytrain,epochs=15, validation_data=(Xtest, ytest))
test_loss, test_acc = model.evaluate(Xtest,  ytest, verbose=2)
print("accuracy ",test_acc)
model.save("Hand Number recognition.model")