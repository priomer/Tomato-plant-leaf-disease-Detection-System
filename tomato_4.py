# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:24:18 2020

@author: ocn
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image

# for plotting images (optional)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    

# getting data
base_dir = 'E:/AVRN_Report/Plant_Diseases_Dataset'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

train_Tomato___Bacterial_spot = os.path.join(train_dir, 'Tomato___Bacterial_spot')
train_Tomato___Early_blight = os.path.join(train_dir, 'Tomato___Early_blight')
train_Tomato___healthy = os.path.join(train_dir, 'Tomato___healthy')
train_Tomato___Late_blight = os.path.join(train_dir, 'Tomato___Late_blight')
train_Tomato___Septoria_leaf_spot = os.path.join(train_dir, 'Tomato___Septoria_leaf_spot')
train_Tomato___Spider_mites = os.path.join(train_dir, 'Tomato___Spider_mites Two-spotted_spider_mite')
train_Tomato___Target_Spot = os.path.join(train_dir, 'Tomato___Target_Spot')
train_Tomato___Tomato_mosaic_virus = os.path.join(train_dir, 'Tomato___Tomato_mosaic_virus')
train_Tomato___Tomato_Yellow_Leaf_Curl_Virus = os.path.join(train_dir, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus')



valid_Tomato___Bacterial_spot = os.path.join(valid_dir, 'Tomato___Bacterial_spot')
valid_Tomato___Early_blight = os.path.join(valid_dir, 'Tomato___Early_blight')
valid_Tomato___healthy = os.path.join(valid_dir, 'Tomato___healthy')
valid_Tomato___Late_blight = os.path.join(valid_dir, 'Tomato___Late_blight')
valid_Tomato___Septoria_leaf_spot = os.path.join(valid_dir, 'Tomato___Septoria_leaf_spot')
valid_Tomato___Spider_mites = os.path.join(valid_dir, 'Tomato___Spider_mites Two-spotted_spider_mite')
valid_Tomato___Target_Spot = os.path.join(valid_dir, 'Tomato___Target_Spot')
valid_Tomato___Tomato_mosaic_virus = os.path.join(valid_dir, 'Tomato___Tomato_mosaic_virus')
valid_Tomato___Tomato_Yellow_Leaf_Curl_Virus = os.path.join(valid_dir, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus')

num_Bacterial_spot_tr = len(os.listdir(train_Tomato___Bacterial_spot))
num_Early_blight_tr = len(os.listdir(train_Tomato___Early_blight))
num_healthy_tr = len(os.listdir(train_Tomato___healthy))
num_Late_blight_tr = len(os.listdir(train_Tomato___Late_blight))
num_Septoria_leaf_tr = len(os.listdir(train_Tomato___Septoria_leaf_spot))
num_Spider_mite_tr = len(os.listdir(train_Tomato___Spider_mites))
num_Target_spot_tr = len(os.listdir(train_Tomato___Target_Spot))
num_mosaic_virus_tr = len(os.listdir(train_Tomato___Tomato_mosaic_virus))
num_yellow_leaf_tr = len(os.listdir(train_Tomato___Tomato_Yellow_Leaf_Curl_Virus))


num_Bacterial_spot_vl = len(os.listdir(valid_Tomato___Bacterial_spot))
num_Early_blight_vl = len(os.listdir(valid_Tomato___Early_blight))
num_healthy_vl = len(os.listdir(valid_Tomato___healthy))
num_Late_blight_vl = len(os.listdir(valid_Tomato___Late_blight))
num_Septoria_leaf_vl = len(os.listdir(valid_Tomato___Septoria_leaf_spot))
num_Spider_mite_vl = len(os.listdir(valid_Tomato___Spider_mites))
num_Target_spot_vl = len(os.listdir(valid_Tomato___Target_Spot))
num_mosaic_virus_vl = len(os.listdir(valid_Tomato___Tomato_mosaic_virus))
num_yellow_leaf_vl = len(os.listdir(valid_Tomato___Tomato_Yellow_Leaf_Curl_Virus))



total_train = num_Bacterial_spot_tr + num_Early_blight_tr + num_healthy_tr + num_Late_blight_tr + num_Septoria_leaf_tr + num_Spider_mite_tr + num_Target_spot_tr + num_mosaic_virus_tr + num_yellow_leaf_tr

total_val = num_Bacterial_spot_vl + num_Early_blight_vl + num_healthy_vl + num_Late_blight_vl + num_Septoria_leaf_vl + num_Spider_mite_vl + num_Target_spot_vl + num_mosaic_virus_vl + num_yellow_leaf_vl 

BATCH_SIZE = 32
IMG_SHAPE = 200 # square image


#generators

#prevent memorization
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

validation_image_generator = ImageDataGenerator(
    rescale=1./255)


train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=valid_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')
images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(images)


model = Sequential()
# Conv2D : Two dimentional convulational model.
# 32 : Input for next layer
# (3,3) convulonational windows size
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SHAPE, IMG_SHAPE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5)) # 1/2 of neurons will be turned off randomly
model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
# output dense layer; since thenumbers of classes are 10 here so we need to pass minimum 10 neurons whereas 2 in cats and dogs   
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


EPOCHS = 10

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )


# analysis
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save("model_tomato_plant_disease.h5")
print("Saved model to disk")