from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
import shutil
import os



#80 images for each class. Class 0 data is from 0-79, Class 1 is 80-159, etc

j = 0
total = 1361
for i  in range(1, total):
    fpath = "17flowers/jpg/image_" + str(i).zfill(4) + ".jpg"
    destPath = 'flower_dataset/' +str(j).zfill(2)
    if not os.path.exists(destPath):
        os.makedirs(destPath)
    shutil.copy(fpath, destPath)

    if i%80==0:
        j+=1



batch_size = 16

#Data Augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)


#Normalize
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'flower_dataset',  
        target_size=(224, 224),  
        batch_size=batch_size,
        class_mode='categorical',
        subset="training")  

validation_generator = train_datagen.flow_from_directory(
        'flower_dataset',  
        target_size=(224, 224),  
        batch_size=batch_size,
        class_mode='categorical',
        subset="validation") 


#Fine tune a pretained VGG16 model

def vgg16_model():
    vgg_conv = VGG16(weights= "imagenet" , include_top=False, 
                     input_shape=(224, 224, 3))
    vgg_conv.trainable = False
    model = Sequential()
    
    for layer in vgg_conv.layers:
        model.add(layer)

    # Add new layers
    model.add(Flatten())
    model.add(Dropout(0.3)) 
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(17, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = vgg16_model()
model.summary()


model.fit_generator(
        train_generator,
        steps_per_epoch = 1088 // batch_size,
        epochs = 5,
        validation_data = validation_generator)