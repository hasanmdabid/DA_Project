'''######################################
    Imports
#######################################'''
from __future__ import print_function
import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from save_result import *
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from data_preprocessing import *
from Deap.unnecessary.data_preprocessing_withscalling import *
from models import *
import Deap.unnecessary.seg_TSAUG as seg_TSAUG
#from vis.backprop_modifiers import get


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
x_train_raw, x_test, y_train_raw, y_test = train_test_split(combined_data_valence_slided, combined_valence_slided, test_size=0.25, random_state=100, stratify=combined_valence_slided)

#**********************Select the Augmntation factor***************
Aug_factor = 1
#******************************************************************
print('Before augmentation sape of X_train and y_train:', x_train_raw.shape, y_train_raw.shape)
x_train, y_train = seg_TSAUG.seg_TSAUG(x_train_raw, y_train_raw, Aug_factor)
print('After augmentation sape of X_train and y_train:', x_train.shape, y_train.shape)

method = 'Cropping'
activation='relu'
init_mode='glorot_uniform'
optimizer='Adam'
dropout_rate=0.6
n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], 1
batch_size = 32
epochs = 500
verbose = 1


def print_info(info):
    print("\n")
    print('-----------------------------------------------------------------------------')
    print(info)
    print('-----------------------------------------------------------------------------')
    print("\n")

def createFitEvaluateModel(modelName, x_train, y_train, x_test, y_test, verbose, epochs, batch_size):

    callbacks_list = []
    
    print_info("Create model")
    if modelName == 'conv1D':
        model=conv1D(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs)
    elif modelName == 'conv2D':
        model=conv2D(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs)
    elif modelName == 'deepconvlstm': 
        model=deepconvlstm(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs)

    print_info("Model created")
    print(model.summary())
      
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50) 
    #mc = ModelCheckpoint("best_model_"+modelName+".h5", monitor='val_accuracy', mode='max', verbose = verbose, save_best_only=True)
    print_info("Fit model:")
    
    #history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    # load the saved model
    #saved_model = load_model("best_model_"+modelName+".h5")

    # Fit the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(x_test, y_test), callbacks=es)
    
    #evaluate the model
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))
    print_info("Evaluate model: ")

   
    return train_acc, test_acc, model



def test_model(modelName, x_train, y_train, x_test, y_test, verbose, epochs, batch_size):

    print(modelName)
    train_acc, test_acc, model = createFitEvaluateModel(modelName, x_train, y_train, x_test, y_test, verbose =verbose, epochs = epochs, batch_size = batch_size)
    predictions = (model.predict(x_test) > 0.5).astype("int32")
    average_fscore_macro = (f1_score(y_test, predictions, average="macro"))
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print(classification_report(y_test, predictions))
    print('\n')

    saveResultsCSV(method, Aug_factor, modelName, epochs, batch_size, train_acc, test_acc, average_fscore_macro)


def test_all_models(x_train, y_train, x_test, y_test, verbose, epochs, batch_size):
    ''' Tests all available networks'''

    modelNames = ['conv1D', 'conv2D']

    for modelName in modelNames:
      
        test_model(modelName, x_train, y_train, x_test, y_test, verbose=verbose, epochs = epochs, batch_size = batch_size)
        keras.backend.clear_session()


test_all_models(x_train, y_train, x_test, y_test, verbose, epochs, batch_size)

