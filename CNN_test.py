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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    
                
                
            
    

