import gc
import numpy as np
from sklearn.metrics import f1_score

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tensorflow as tf
from data_preprocessing import *
from data_preprocessing_withscalling import *
from models import *
from save_result import *
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
import seg_TSAUG
from DTW import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# -------------------------------------------Importing the preprocessed data----------------
eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided = data_pre_pro_without_scalling()

print('Shape of eeg_valence_data:', eeg_valence_slided.shape)
print('Shape of valence_labels:', valence_slided.shape)
print('Shape o eeg_arousal_data:', eeg_arousal_slided.shape)
print('Shape of arousal_labels:', arousal_slided.shape)
print('Shape of combined_data_valence_slided:', combined_data_valence_slided.shape)
print('Shape of combined_valence_labels:', combined_valence_slided.shape)
print('Shape of combined_data_arousal_slided:', combined_data_arousal_slided.shape)
print('Shape of combined_arousal_labels:', combined_arousal_slided.shape)



# Considering only Valence 
method = 'Randomtimewarp'
activation='relu'
init_mode='glorot_uniform'
optimizer='Adam'
dropout_rate=0.6
batch_size = 32
epochs = 200
verbose = 2
modelName = 'CONV2D'
Aug_factor = 1

Family_name = input('Mention the Family name ("TSAUG, "TW", "RTW", DTW"):') 


x_train_raw, x_test, y_train_raw, y_test = train_test_split(combined_data_valence_slided, combined_valence_slided, test_size=0.25, random_state=100, stratify=valence_slided)
for i in range(1, 3):
    Aug_factor = i
    if Aug_factor == 0:
        x_train = x_train_raw
        y_train = y_train_raw
    else:
        if Family_name == 'TSAUG':
            x_train, y_train = seg_TSAUG.seg_TSAUG(x_train_raw, y_train_raw, Aug_factor)
        elif Family_name == 'DTW':
            method = 'GuidedTimeWarp'
            if Aug_factor == 1:
                guided_DTW_1, labels = DTW(x_train_raw, y_train_raw)
                x_train = np.concatenate((x_train_raw, guided_DTW_1), axis=0)
                y_train = np.concatenate((y_train_raw, labels), axis=0)
            elif Aug_factor == 2:
                guided_DTW_1, labels = DTW(x_train_raw, y_train_raw)
                guided_DTW_2, labels = DTW(x_train_raw, y_train_raw)
                x_train = np.concatenate((x_train_raw, guided_DTW_1, guided_DTW_2), axis=0)
                y_train = np.concatenate((y_train_raw, labels, labels), axis=0)
        elif Family_name == 'RTW':
            method = 'RandomTimeWarp'
            if Aug_factor == 1:
                guided_DTW_1, labels = RTW(x_train_raw, y_train_raw)
                x_train = np.concatenate((x_train_raw, guided_DTW_1), axis=0)
                y_train = np.concatenate((y_train_raw, labels), axis=0)
            elif Aug_factor == 2:
                guided_DTW_1, labels = RTW(x_train_raw, y_train_raw)
                guided_DTW_2, labels = RTW(x_train_raw, y_train_raw)
                x_train = np.concatenate((x_train_raw, guided_DTW_1, guided_DTW_2), axis=0)
                y_train = np.concatenate((y_train_raw, labels, labels), axis=0)
        elif Family_name == 'TW':
            method = 'TimeWarp'
            if Aug_factor == 1:
                guided_DTW_1, labels = TW(x_train_raw, y_train_raw)
                x_train = np.concatenate((x_train_raw, guided_DTW_1), axis=0)
                y_train = np.concatenate((y_train_raw, labels), axis=0)
            elif Aug_factor == 2:
                guided_DTW_1, labels = TW(x_train_raw, y_train_raw)
                guided_DTW_2, labels = TW(x_train_raw, y_train_raw)
                x_train = np.concatenate((x_train_raw, guided_DTW_1, guided_DTW_2), axis=0)
                y_train = np.concatenate((y_train_raw, labels, labels), axis=0)
            
        
    print('Shape of x_train, y_train:', x_train.shape, y_train.shape)
    #-----------------------------------------------------Evaluating the Model By usnig the Model check point----------------------------------------------------------------------------------------------------
    # # Fitting the model With the system.
    seed = 7
    tf.random.set_seed(seed)
    # Initialze the estimators
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], 1
    model = conv2D(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs)
    print("Fit model:")
    #-----------------------------------------------------------------Using early stop and Model Check point------------------------------------------------------------------------
    # # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=verbose, callbacks=es)
    # load the saved model
    #saved_model = load_model('best_model.h5')
    # evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))

    predictions = (model.predict(x_test) > 0.5).astype("int32")
    average_fscore_macro = (f1_score(y_test, predictions, average="macro"))
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print(classification_report(y_test, predictions))
    print('\n')
    saveResultsCSV(method, Aug_factor, modelName, epochs, batch_size, train_acc, test_acc, average_fscore_macro) # In save results we are providing the AUG factor
    del x_train, y_train
    tf.keras.backend.clear_session()
    gc.collect()

  

"""
#-----------------------------------------------------Performing the Gridsearch-cv -----------------------------------------------------------------------------------------
# fix random seed for reproducibility

seed = 7
tf.random.set_seed(seed)


# Initialze the estimators
activation='relu'
init_mode='glorot_uniform'
optimizer='Adam'
dropout_rate=0.6
n_timesteps, n_features, n_outputs = combined_data_valence_slided.shape[1], combined_data_valence_slided.shape[2], 1
model1 = KerasClassifier(model=conv1D(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs), verbose=0)

batch_size = [10, 20, 50, 60]
epochs = [10, 50, 100]
dropout_rate = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

estimator= model1
param_grid_1st_evaluation = dict(batch_size=batch_size, epochs= epochs)
param_grid_2nd_Evaluation = dict(epochs=epochs, batch_size=batch_size, model__dropout_rate=dropout_rate, model__activation=activation, optimizer=optimizer, model__init_mode=init_mode)
grid = GridSearchCV(estimator= estimator, param_grid=param_grid_1st_evaluation, scoring="accuracy", refit="accuracy", cv=3)
grid_result = grid.fit(x_train, y_train)
# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#print('Best Estimator :\n', grid.best_estimator_)      # To get the best estimator
print('Best Parameter :\n', grid.best_params_)         # to get the best parameters
print('Best Score :\n', grid.best_score_)

"""

"""
#================================================================ Model Selection----------------------------------------------------------------------------------------------------
# From the above code, we have found the best model (CONV2D).
# THe highest accuracy is 86.091%.

# now we will use the best model (CONV2D) with the augmented data to increase the accuracy.


# Fitting the model With the system.

activation='relu'
init_mode='glorot_uniform'
optimizer='Adam'
dropout_rate=0.6
n_timesteps, n_features, n_outputs = combined_data_valence_slided.shape[1], combined_data_valence_slided.shape[2], 1


model = conv2D(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs)
print("Fit model:")

method = 'WithoutAugmentation'
modelname = 'Conv2D'
epochs = 250
batch_size = 32

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
predictions = (model.predict(x_test) > 0.5).astype("int32")
average_fscore_macro = (f1_score(y_test, predictions, average="macro"))
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))
print('\n')
#time_elapsed = datetime.now() - start_time

saveResultsCSV(method, modelname, epochs, batch_size, accuracy, average_fscore_macro, "", "", "")

del x_train, x_test, y_test, y_train, eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided
tf.keras.backend.clear_session()
gc.collect()

"""

