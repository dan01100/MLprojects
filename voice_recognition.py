from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Activation, Dropout, Flatten, Dense, BatchNormalization, MaxPool2D
from sklearn.model_selection import train_test_split
import librosa
import os
import json
import numpy as np


categories = ["zero", "one", "two", "three", "four", 
              "five", "six", "seven", "eight", "nine"]

#categories = ["zero", "one"]

NUM_SAMPLES = 22050

labels = []
data = []

#Getting data and labels

for i, category in enumerate(categories):
    directory = "speech_dataset/" + category
    
    for filename in os.listdir(directory):
        signal, sr = librosa.load(os.path.join(directory, filename))
        
        if len(signal) >= NUM_SAMPLES:
        
            MFFCs = librosa.feature.mfcc(signal[:NUM_SAMPLES], n_mfcc=13, hop_length=512, n_fft=2048)
            labels.append(i)
            data.append(MFFCs.T)
            print(filename)
        
        
        
labels = np.asarray(labels)
data = np.asarray(data)


#Prepare training and testing data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

#train model

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(44, 13, 1), kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=(2, 2), padding='same'))

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=(2, 2), padding='same'))

model.add(Conv2D(16, kernel_size=(2, 2),
                 activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile('adam', 'sparse_categorical_crossentropy', metrics=["acc"])

model.summary()


model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test, y_test))


