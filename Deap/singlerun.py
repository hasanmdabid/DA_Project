import gc

from sklearn.metrics import f1_score, accuracy_score

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

import tensorflow as tf

from data_preprocessing import *
from models import *
from save_result import *

import seg_TSAUG

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# -------------------------------------------Importing the preprocessed data----------------
eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided = data_pre_pro_with_scalling()

print('Shape of eeg_valence_data:', eeg_valence_slided.shape)
print('Shape of valence_labels:', valence_slided.shape)
print('Shape o eeg_arousal_data:', eeg_arousal_slided.shape)
print('Shape of arousal_labels:', arousal_slided.shape)
print('Shape of combined_data_valence_slided:',combined_data_valence_slided.shape)
print('Shape of combined_valence_labels:', combined_valence_slided.shape)
print('Shape of combined_data_arousal_slided:',combined_data_arousal_slided.shape)
print('Shape of combined_arousal_labels:', combined_arousal_slided.shape)


# Considering only Valence
method = 'None'
activation = 'relu'
init_mode = 'glorot_uniform'
optimizer = 'Adam'
dropout_rate = 0.6
batch_size = 32
epochs = 100
verbose = 2
modelName = 'CONV2D'
Aug_factor = "None"

x_train_raw, x_test, y_train_raw, y_test = train_test_split(combined_data_valence_slided, combined_valence_slided, test_size=0.2, random_state=100, stratify=valence_slided)

print(f"Training data shape: x = {x_train_raw.shape}, y= {y_train_raw.shape}")

#**********************************************Checking the Results Without AugmentatioN****************************************
x_train = x_train_raw
y_train = y_train_raw
#**********************************************Checking the Augmentation For Tsaug***************************************** 
#x_train, y_train = seg_TSAUG.seg_TSAUG(x_train_raw, y_train_raw, Aug_factor)
#print('Training data shape:', x_train.shape)

# -----------------------------------------------------Evaluating the Model By usnig the Model check point----------------------------------------------------------------------------------------------------
n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], 1
model = conv2D(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs)
print("Fit model:")
# -----------------------------------------------------------------Using early stop and Model Check point------------------------------------------------------------------------
model.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=epochs, verbose=verbose)
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Train Accuracy: %.3f, Test Accuracy: %.3f' %(train_acc, test_acc))

predictions = (model.predict(x_test) > 0.5).astype("int32")
best_f1score_macro = (f1_score(y_test, predictions, average="macro"))
best_accuracy = accuracy_score(y_test, predictions)
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))
print('\n')
saveResultsCSV(method, Aug_factor, modelName, epochs, batch_size, train_acc, test_acc, best_f1score_macro, best_accuracy)

gc.collect()
