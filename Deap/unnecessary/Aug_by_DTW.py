import gc
import numpy as np
from sklearn.metrics import f1_score
import math
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import slided_numpy_array
from keras.models import load_model
from sklearn.model_selection import train_test_split

import tensorflow as tf
from data_preprocessing import *

from models import *
from save_result import *
from sklearn.utils import resample
from DTW import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



# -------------------------------------------Importing the preprocessed data----------------
eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided = data_pre_pro_with_scalling()








# -------------------------------------------Importing the preprocessed data----------------
eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided = data_pre_pro_with_scalling()

print('Shape of eeg_valence_data:', eeg_valence_slided.shape)
print('Shape of valence_labels:', valence_slided.shape)
print('Shape o eeg_arousal_data:', eeg_arousal_slided.shape)
print('Shape of arousal_labels:', arousal_slided.shape)
print('Shape of combined_data_valence_slided:',combined_data_valence_slided.shape)
print('Shape of combined_valence_labels:', combined_valence_slided.shape)
print('Shape of combined_data_arousal_slided:', combined_data_arousal_slided.shape)
print('Shape of combined_arousal_labels:', combined_arousal_slided.shape)


# Considering only Valence
activation = 'relu'
init_mode = 'glorot_uniform'
optimizer = 'Adam'
dropout_rate = 0.6
batch_size = 32
epochs = 200
verbose = 2
modelName = 'CONV2D'
Aug_factor = 1

x_train_raw, x_test, y_train_raw, y_test = train_test_split(combined_data_valence_slided, combined_valence_slided, test_size=0.25, random_state=100, stratify=valence_slided)
print('X_train data shape Before Augmentation:', x_train_raw.shape)
print('Y_train data shape Before Augmentation:', y_train_raw.shape)


#***************To use the DTW algorithm the shape of the input is x = 3D(Nr. segents, row, column) 
#***************y = 1D(Nr segments or labels) and Augmentaion factor (Integer value) 
Family_name = input('Mention the Family name ("TW", "RTW", DTW"):')
if Family_name == 'DTW':
      x_aug, y_aug = DTW(x_train_raw, y_train_raw)
      method = 'GuidedTimeWarp'
elif Family_name == 'RTW':
      x_aug, y_aug = RTW(x_train_raw, y_train_raw)
      method = 'Randomtimewarp'
elif Family_name == 'TW':
      x_aug, y_aug = TW(x_train_raw, y_train_raw)
      method = 'TimeWarp'

#***************To use the TSAUG algorithm the shape of the output is x = 3D(Nr. segents, row, column) 
#***************y = 1D(Nr segments or labels)

print('X_Augmentation:', x_aug.shape) # Shape of x_aug is 3D
print('Y_Augmentation:', y_aug.shape) # Shape of y_aug is 1D

x_aug_2d = x_aug.reshape(x_aug.shape[0]*x_aug.shape[1], x_aug.shape[2])  #Converting data from 3D to 2D
y_aug_2d = np.expand_dims(y_aug, axis=1)                                 # Adding a dimension to convert it FROM 1D to 2D
y_aug_2d = np.repeat(y_aug_2d, 128, axis=0)                              # Converting the labels into total number (128) of time stamp


# First we will convert the 3D array to 2D arrays.  Then we will add the labels infromation at the last column to form a 2D Augmented datset.
Aug_data = np.concatenate((x_aug_2d, y_aug_2d), axis=1) 
print('Shape of Augmented dataset:', Aug_data.shape) # Shape is 2D



for i in np.arange(0.2, 1, 0.2):
      
      Aug_frac = i  # Select the values of the augmatation fraction.   
      n_ecp_samples = math.ceil(Aug_data.shape[0]*Aug_frac) 
      print('Shape of Number of Expected samples:', n_ecp_samples)

      #---------------------------------------------------------------- Performing the Downsample With SKlearn----------------------------------------------------------------
      # We applied resample() method from the sklearn.utils module for downsampling, The replace = True attribute performs random resampling with replacement. The n_samples attribute 
      # defines the number of records you want to select from the original records. We have set the value of this attribute to the number of records in the spam dataset so the two sets will be balanced.
      Aug_downsample = resample(Aug_data, replace=True, n_samples=n_ecp_samples, random_state=42)

      print(Aug_downsample.shape)

      #---------------------------------------------------Converting the Labels into 1D array------------------------------------------------
      # This part of code will 1st selec the Number of samples
      Aug_frac_x, Aug_frac_y = slided_numpy_array.slided_numpy_array(Aug_downsample)
      print('Shape of training Augmented data with fractional amount:',Aug_frac_x.shape)
      print('Shape of training Augmented data with fractional amount:',Aug_frac_y.shape)

      x_train = np.concatenate((x_train_raw,Aug_frac_x), axis=0)
      y_train = np.concatenate((y_train_raw,Aug_frac_y), axis=0)
      print('X_train data shape after fractional Augmentation:', x_train.shape)
      print('Y_train data shape after fractional Augmentation:', y_train.shape)
      

      # -----------------------------------------------------Evaluating the Model By usnig the Model check point----------------------------------------------------------------------------------------------------
      seed = 7
      tf.random.set_seed(seed)
      n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], 1    
      model = conv2D(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs)
      es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
      print("Fit model:")
      # -----------------------------------------------------------------Using early stop and Model Check point------------------------------------------------------------------------
      model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=verbose, callbacks=es)
      _, train_acc = model.evaluate(x_train, y_train, verbose=0)
      _, test_acc = model.evaluate(x_test, y_test, verbose=0)
      print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))

      predictions = (model.predict(x_test) > 0.5).astype("int32")
      average_fscore_macro = (f1_score(y_test, predictions, average="macro"))
      print(confusion_matrix(y_test, predictions))
      print('\n')
      print(classification_report(y_test, predictions))
      print('\n')
      saveResultsCSV(method, Aug_frac, modelName, epochs, batch_size, train_acc, test_acc, average_fscore_macro) # In save results we are providing the AUG frac
      del x_train, y_train
      tf.keras.backend.clear_session()
      gc.collect()
