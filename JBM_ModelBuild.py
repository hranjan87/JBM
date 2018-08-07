
# coding: utf-8

# In[59]:


import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras import callbacks
import  sklearn
import sklearn.datasets as skds
import pandas as pd
from keras import backend as K


train_data_dir = '/Users/HimanshuRanjan/MachineLearning/JBM/All_61326/train_61326'
test_data_dir = '/Users/HimanshuRanjan/MachineLearning/JBM/All_61326/test_61326'
validation_data_dir = '/Users/HimanshuRanjan/MachineLearning/JBM/All_61326/validation_61326'
img_width, img_height = 150, 150
classes_num = 6
lr = 0.0004
nb_train_samples = 160
nb_validation_samples = 20
epochs = 20
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes_num))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')



# In[60]:


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# In[61]:


target_dir = '/Users/HimanshuRanjan/MachineLearning/JBM/models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('/Users/HimanshuRanjan/MachineLearning/JBM/models/model.h5')
model.save_weights('/Users/HimanshuRanjan/MachineLearning/JBM/models/weights.h5')


# In[62]:


score = model.evaluate_generator(validation_generator, nb_validation_samples/batch_size, workers=12)


# In[63]:


Accuracy = score[1]*100


# In[64]:


print(Accuracy)

