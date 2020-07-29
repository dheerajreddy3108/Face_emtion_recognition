# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:40:09 2020

@author: dheeraj_reddy_peram
"""

#Import all required libraries in the first step
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,BatchNormalization,Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import math
from keras import utils as k
from keras.callbacks import EarlyStopping,ReduceLROnPlateau, Callback

#Reading the dataset file
dataset = pd.read_csv('fer2013.csv')

#checking dataset shape and checking unique emotions in given dataset
print(dataset.shape)
unique = dataset.emotion.unique()
print(unique)

#labelling the emotions as given by the dataset 
emotion_encoder = {0:'anger',1:'disgust',2:'fear',3:'happy',4:'sadness',5:'surprise',6:'neutral'}

#checking number of images for each emotion
count = dataset.emotion.value_counts()
#checking size of image
size = math.sqrt(len(dataset.pixels[0].split(' ')))


img = plt.figure(1,(14,14))

a = 0

for label in sorted(dataset.emotion.unique()):
    for i in range(0,7):
        xp = dataset[dataset.emotion == label].pixels.iloc[a]
        xp= np.array(xp.split(' ')).reshape(48,48).astype('float32')
        a = a+1
        ax = plt.subplot(7,7,a)
        ax.imshow(xp,cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(emotion_encoder[label])
        plt.tight_layout()
        
#Datapreprocessing 
img_ar = dataset.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48,48,1).astype('float32'))
img_ar = np.stack(img_ar, axis = 0)
img_ar.shape

#Image Encoding
label_encoder = LabelEncoder()
img_labels = label_encoder.fit_transform(dataset.emotion)
img_labels = k.to_categorical(img_labels)

#Data Splitting

X_train,X_test,y_train,y_test = train_test_split(img_ar,img_labels,test_size=0.2,shuffle=True,random_state=123)

#Image Shape 
img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
classes_count= y_train.shape[1]

#Normalization
X_train = X_train/255.
X_test= X_test/255.

#Building CNN
model = Sequential()
model.add(Conv2D(128,(5,5),input_shape=(img_width,img_height,img_depth),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(classes_count,activation='softmax'))
model.summary()

#model Compile
model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stopping=EarlyStopping(monitor='val_accuracy', min_delta=0.00001,patience=11,verbose=1,restore_best_weights=True)
reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,verbose=1)
callbaks= [early_stopping,reduce_lr]

#Image Augmentation
train_datagen=ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True)

train_data = train_datagen.fit(X_train)

history = model.fit_generator(train_datagen.flow(X_train,y_train,batch_size=32),
                              validation_data = (X_test,y_test),
                              steps_per_epoch=len(X_train)/64,
                              epochs=100,verbose=1)

model.save('model_emotion.h5')

y_valid = model.predict(X_test)

