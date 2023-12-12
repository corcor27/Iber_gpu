import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

forward_lstm = LSTM(units=320, return_sequences=True)
# backward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
brnn = Bidirectional(forward_lstm)

print('building model')

model = Sequential()
model.add(Convolution1D(activation="relu", 
                        input_shape=(1000, 4), 
                        padding="valid", strides=1, 
                        filters=320, kernel_size=26))

model.add(MaxPooling1D(strides=13, pool_size=13))

model.add(Dropout(0.2))

model.add(brnn)

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, units=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, units=919))
model.add(Activation('sigmoid'))

print('compiling model')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['cosine'])

print('running at most 60 epochs')

checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.summary()
    
          
    
                
                
            
    

