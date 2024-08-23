# pylint: disable-all

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from keras.callbacks import EarlyStopping
import platform
import tensorflow as tf
from data_preprocessing import *
from models import *
from save_result import *
from augmenter import*
from DTW import *
from sklearn.model_selection import StratifiedKFold

random.seed(50)
np.random.seed(50)


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


def check_gpu():

    if 'linux' in platform.platform().lower():
        print("Check GPU...")
        if len(tf.config.list_physical_devices('GPU')) == 0:
            print("GPU is not available!")
            quit()

        print("GPU is available!")

check_gpu()

# -------------------------------------------Importing the preprocessed data----------------
eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided = data_pre_pro_with_scalling()

print('Shape of eeg_valence_data:', eeg_valence_slided.shape)                        #(80640, 128, 32)
print('Shape of valence_labels:', valence_slided.shape)                              #(80640,)
print('Shape o eeg_arousal_data:', eeg_arousal_slided.shape)                         #(80640, 128, 32)
print('Shape of arousal_labels:', arousal_slided.shape)                              #(80640,) 
print('Shape of combined_data_valence_slided:', combined_data_valence_slided.shape)  #(80640, 128, 40)
print('Shape of combined_valence_labels:', combined_valence_slided.shape)            #(80640,)
print('Shape of combined_data_arousal_slided:', combined_data_arousal_slided.shape)  #(80640, 128, 40)
print('Shape of combined_arousal_labels:', combined_arousal_slided.shape)            #(80640,)
# ["jitter", "scaling", "rotation",  "slicing", "permutation", "magnitude_warp",  "TW", "WW", "RGW", "DGW", "spawner", "cGAN" ]
 
param = {"aug_method": ["jitter", "scaling", "rotation",  "slicing", "permutation", "magnitude_warp",  "TW", "WW", "RGW", "DGW", "spawner", "cGAN"],
            "aug_factor" : [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4],
            "activation" : 'relu', 
            "init_mode" : 'glorot_uniform',
            "optimizer" : 'Adam',
            "dropout_rate" : 0.4,
            "batch_size" : 32,
            "epochs" : 100,
            "verbose" : 2,
            "modelname" : 'CONV2D'          
         }

# Select  the data to be used for the convolution algorithm 
label_name = input('Which Label do you want to use for the computation? valence or arousal\n')
if label_name == "valence":
    x = combined_data_valence_slided
    y = combined_valence_slided
elif label_name == "arousal":
    x = combined_data_arousal_slided
    y = combined_arousal_slided

max_val_sacled = np.max(x)
min_val_scaled = np.min(x)
print('Maximume_value of Raw DATA:', max_val_sacled)
print('Minimume_value of Raw DATA:', min_val_scaled)


#x_train_raw, x_test, y_train_raw, y_test = train_test_split(x, y, test_size=0.25, random_state=100, stratify=valence_slided)
#print(f"Shape of x_train_raw {x_train_raw.shape} and y_trai_raw {y_train_raw.shape}")


def conduct_experiment(x_train, y_train, x_test, y_test, activation = param["activation"], init_mode = param["init_mode"], optimizer = param["optimizer"], dropout_rate = param["dropout_rate"], 
                       epochs = param["epochs"], verbose = param["verbose"], batch_size = param["batch_size"]):
    
    seed = 7
    tf.random.set_seed(seed)
    # Initialze the estimators
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], 1
    model = conv2D(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs)
    print("Fit model:")
    #-----------------------------------------------------------------Using early stop and Model Check point------------------------------------------------------------------------
    # # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=100)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), epochs=epochs, verbose=verbose, callbacks=es)
    # load the saved model
    #saved_model = load_model('best_model.h5')
    # evaluate the model
    
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))
    predictions = (model.predict(x_test) > 0.5).astype("int32")
    f1score_macro = (f1_score(y_test, predictions, average="macro"))
    accuracy = accuracy_score(y_test, predictions)
    tf.keras.backend.clear_session()
    gc.collect()
    
    return train_acc,test_acc, accuracy, f1score_macro


# Iterate over each aug_method
for aug_method in param["aug_method"]:
    # Iterate over each aug_factor for the current aug_method
    for aug_factor in param["aug_factor"]:
        try:
            print(f"Augmentation method is {aug_method} and augmentation factor is {aug_factor}")
            # Define the number of folds
            n_splits = 5  # You can adjust this number

            # Initialize K-fold cross-validator
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Initialize variables to store results
            best_accuracy = 0
            best_f1_score = 0
            best_train_accuracy = 0
            best_test_accuracy = 0
            all_accuracies = []
            all_f1_scores = []
            
            for train_index, test_index in tqdm(kf.split(x, y), total=n_splits, desc="K-Fold Cross-validation"):
                x_train_raw, x_val = x[train_index], x[test_index]
                y_train_raw, y_val = y[train_index], y[test_index]
                print(f"Shape of x_train_raw {x_train_raw.shape}, y_train_raw {y_train_raw.shape}")
                
                # Augment the data
                x_aug, y_aug = augment(aug_factor, aug_method, label_name, x_train_raw, y_train_raw)
                
                print(f"Shape of x_aug train is {x_aug.shape}, y_aug train is {y_aug.shape}")
                x_train = np.concatenate((x_train_raw, x_aug), axis=0)
                y_train = np.concatenate((y_train_raw, y_aug), axis=0)
                print('Shape of x_train, y_train:', x_train.shape, y_train.shape)
                #-----------------------------------------------------Conduct the Experiment----------------------------------------------------------------------------------------------------
                train_acc,test_acc, accuracy, f1score_macro =conduct_experiment(x_train, y_train, x_test=x_val, y_test=y_val, activation = param["activation"], 
                                                                                init_mode = param["init_mode"], optimizer = param["optimizer"], dropout_rate = param["dropout_rate"],
                                                                                 epochs = param["epochs"], verbose = param["verbose"], batch_size = param["batch_size"])
                
                all_accuracies.append(accuracy)
                all_f1_scores.append(f1score_macro)
                
                 # Update best train and test accuracy                
                if train_acc > best_train_accuracy:
                    best_train_accuracy = train_acc
                if test_acc > best_test_accuracy:
                    best_test_accuracy = test_acc

                # Update best accuracy and F1 score
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                if f1score_macro > best_f1_score:
                    best_f1_score = f1score_macro
            #Calculate the average accuracy and f1 score. 
            average_accuracy = np.mean(all_accuracies)
            average_f1_score = np.mean(all_f1_scores)
            # Calculate standard deviation of average accuracy and average F1 score
            std_accuracy = np.std(all_accuracies)
            std_f1_score = np.std(all_f1_scores)
                
            
            saveResultsCSV(label=label_name, aug_method=aug_method, aug_factor=aug_factor, modelname=param["modelname"],
                            epochs=param["epochs"], batch_size=param["batch_size"], test_acc=best_test_accuracy,
                            best_f1score_macro=best_f1_score, avg_f1score=average_f1_score, std_f1score=std_f1_score,
                            best_accuracy=best_accuracy, avg_acc=average_accuracy, std_acc=std_accuracy,
                            all_accuracies=all_accuracies, all_f1_scores=all_f1_scores) 
            
            del x_train, y_train, x_aug, y_aug, all_accuracies, all_f1_scores
            gc.collect()
            
        except Exception as e:
            print(e)
            