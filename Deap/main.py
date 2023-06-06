import gc
import pandas as pd
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D


from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.constraints import MaxNorm
from data_preprocessing import *
from data_preprocessing_withscalling import *
from models import *
from save_result import *
from seg_SMOTE import *
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# -------------------------------------------Importing the preprocessed data----------------
"""
eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided = data_pre()

print('Shape of eeg_valence_data:', eeg_valence_slided.shape)
print('Shape of valence_labels:', valence_slided.shape)
print('Shape o eeg_arousal_data:', eeg_arousal_slided.shape)
print('Shape of arousal_labels:', arousal_slided.shape)
print('Shape of combined_data_valence_slided:', combined_data_valence_slided.shape)
print('Shape of combined_valence_labels:', combined_valence_slided.shape)
print('Shape of combined_data_arousal_slided:', combined_data_arousal_slided.shape)
print('Shape of combined_arousal_labels:', combined_arousal_slided.shape)
"""


combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided = seg_SMOTE()

# Considering only Valence 
n_timesteps, n_features, n_outputs = combined_data_valence_slided.shape[1], combined_data_valence_slided.shape[2], 1

x_train, x_test, y_train, y_test = train_test_split(combined_data_valence_slided, combined_valence_slided, test_size=0.25, random_state=100, stratify=combined_valence_slided)



"""
#-----------------------------------------------------Performing the Gridsearch-cv -----------------------------------------------------------------------------------------
# fix random seed for reproducibility

seed = 42
tf.random.set_seed(seed)

model1 = KerasClassifier(model=conv2D)
model2 = KerasClassifier(model=conv1D, verbose=0)
model3 = KerasClassifier(model=deepconvlstm, verbose=0)

batch_size = [50, 60, 80, 100]
epochs = [10, 50, 100]
dropout_rate = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

#param_grid = dict(epochs=epochs, batch_size=batch_size, model__dropout_rate=dropout_rate, model__activation=activation, optimizer=optimizer, model__init_mode=init_mode)
param_grid = dict(batch_size=batch_size, model__dropout_rate= dropout_rate)
grid = GridSearchCV(estimator=model3, param_grid=param_grid, scoring="accuracy", refit="accuracy", cv=2)
grid_result = grid.fit(x_train, y_train, epochs=10)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print('Best Estimator :\n', grid.best_estimator_)      # To get the best estimator
print('Best Parameter :\n', grid.best_params_)         # to get the best parameters
print('Best Score :\n', grid.best_score_)
df = pd.DataFrame(grid.cv_results_)
df = df.sort_values("rank_test_accuracy")
df.to_csv('cv_results_batch_epoch.csv')


#----------------------------------------------------------------Create the model----------------------------------------------------------------------------------------------------
# Fitting the model With the system.
model = conv2D()
print("Fit model:")


#-----------------------------------------------------------------Using early stop and Model Check point------------------------------------------------------------------------
# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, verbose=0, callbacks=[es, mc])
# load the saved model
saved_model = load_model('best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(x_train, y_train, verbose=0)
_, test_acc = saved_model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
"""
#================================================================ Model Selection----------------------------------------------------------------------------------------------------
# Fromt the above cod, we have found the best model (CONV2D).
# THe highest accuracy is 86.091%.

# now we will use the best model (CONV2D) with the augmented data to increase the accuracy.
def conv2D(activation='relu', init_mode='glorot_uniform', optimizer='Adam', dropout_rate=0.6):
    model = Sequential()
    # Adding Batch normalization before CONV
    model.add(BatchNormalization(input_shape=(n_timesteps, n_features, 1)))

    # 1st convolutional + pooling
    model.add(Conv2D(filters=256, kernel_size=(3, 1), activation=activation, kernel_initializer=init_mode, input_shape=(n_timesteps, n_features, 1)))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    # 2nd convolutional + pooling + normalization layer
    model.add(Conv2D(filters=128, kernel_size=(3, 1), activation=activation, kernel_initializer=init_mode))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    # 3rd block: convolutional + RELU + normalization
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activation, kernel_initializer=init_mode))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    # 4th block: convolutional + RELU + normalization
    model.add(Conv2D(filters=32, kernel_size=(3, 1), activation=activation, kernel_initializer=init_mode))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))


    # Fully-connected layer
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(32, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # Compiling The model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

model = conv2D()

# Fitting the model With the system.

print("Fit model:")

method = 'SMOTE'
modelname = 'Conv2D'
epochs = 250
batch_size = 32

model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

predictions = (model.predict(x_test) > 0.5).astype("int32")
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
average_fscore_macro = (f1_score(y_test, predictions, average="macro"))
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))
print('\n')
#time_elapsed = datetime.now() - start_time

saveResultsCSV(method, modelname, epochs, batch_size, accuracy, average_fscore_macro, "", "", "")

#del x_train, x_test, y_test, y_train, eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided
del x_train, x_test, y_test, y_train, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided
tf.keras.backend.clear_session()
gc.collect()
