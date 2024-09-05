# pylint: disable-all

"""
In this script we perform Human activities classification on HAR dataset.
To use theis repository for HAR dataset, one must have to download the "HAR" dataset from the following link. 
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
The dataset is public and anyone can use it without any prior consent. 
To read the dataset, please save the csv files under teh following folders "./HAR/data/"
In our study, we used the raw signal from each sensor channels.  
"""


from numpy import dstack
from pandas import read_csv, DataFrame
import numpy as np
from time import time
import os.path
from pathlib import Path
from augmenters import *
import platform
import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM, Lambda, Reshape
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.metrics import f1_score
import gc

# load a single file as a numpy array
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

# load a single file as a numpy array


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]


def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test


def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_' +
                  group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_' +
                  group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_' +
                  group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# load the dataset, returns train and test X and y elements


def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group(
        'train', prefix + '/home/abidhasan/Documents/DA_Project/HAR/data/HARDataset/')
    print('Train 3D data:', trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group(
        'test', prefix + '/home/abidhasan/Documents/DA_Project/HAR/data/HARDataset/')
    print('Test 3D data:', testX.shape, testy.shape)
    return trainX, trainy, testX, testy

# define models


def deepconvlstm(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_channels, n_outputs):
    # First adding the Batch Normalization layer.
    def ReshapeLayer(x):
        shape = x.shape
        reshape = Reshape((shape[1], shape[2] * shape[3]))(x)
        return reshape

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activation,
              kernel_initializer=init_mode, input_shape=(n_timesteps, n_features, n_channels)))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(filters=64, kernel_size=(3, 1),
              activation=activation, kernel_initializer=init_mode))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(filters=64, kernel_size=(3, 1),
              activation=activation, kernel_initializer=init_mode))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(filters=32, kernel_size=(3, 1),
              activation=activation, kernel_initializer=init_mode))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    model.add(Lambda(ReshapeLayer))
    model.add(LSTM(64, activation='tanh', return_sequences=True,
              kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(32, activation='tanh', return_sequences=False,
              kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(32, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(n_outputs, kernel_initializer=init_mode, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model

# save results to csv file


def save_results_to_csv(results, path='/home/abidhasan/Documents/DA_Project/HAR/results/har.csv'):
    if not os.path.exists('/home/abidhasan/Documents/DA_Project/HAR/results/'):
        os.makedirs('/home/abidhasan/Documents/DA_Project/HAR/results/')

    file = Path(path)

    df = DataFrame(results)
    if file.exists():
        df.to_csv(file, mode='a', header=False, index=False, sep=';')
    else:
        df.to_csv(file, index=False, sep=';')
    print(f'Results saved to {file}')


# In this experiment we are considering all the methods except generative
# ['jitter', 'scaling', 'rotation', 'permutation', 'magnitude_warp', 'slicing', 'time_warp', 'window_warp', 'spawner', 'random_guided_warp', 'discriminative_guided_warp', 'cGAN']
factors = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4]
methods = ['jitter', 'scaling', 'rotation', 'permutation', 'magnitude_warp', 'slicing', 'time_warp', 'window_warp', 'spawner', 'random_guided_warp', 'discriminative_guided_warp', 'cGAN']

# load dataset
trainX, trainy, testX, testy = load_dataset()
# converting the train and testy data into 1D
trainy, testy = trainy[:, 0], testy[:, 0]
print('Shape of train_x:{}, train_y:{}, test_x:{}, test_y:{}'.format(
    trainX.shape, trainy.shape, testX.shape, testy.shape))
print('Unique Indices in train set:', np.unique(trainy))
print('Unique indices in test  set:', np.unique(testy))
unique, counts = np.unique(trainy, return_counts=True)
print('Number of instances in train Y:')
print(np.asarray((unique, counts)).T)

# ----------------------- Assign the parameters of the model-------------------------------------------------
# Update n_outputs to 7
nr_samples, n_timesteps, n_features, n_channels = trainX.shape[0], trainX.shape[1], trainX.shape[2], 1
activationConv = 'relu'
activationMLP = 'relu'
verbose = 2
epochs = 300
batch_size = 32
# Specify How many times do you want to rerun the test
num_repeats = 5

# ---------------------Reshape the input according to the model------------------------------------------------
# Fit and Evaluate the Model


def createandevaluate(trainx, testx, trainy, testy, n_timesteps, n_features, n_outputs,  activationConv,  verbose, epochs, batch_size, factor, method):
    modelname = 'DEEPCONVLSTM'
    model = deepconvlstm(activationConv, 'he_normal', 'adam',
                         0.3, n_timesteps, n_features, n_channels, n_outputs)
    print("Model created")

    # Initialize results dictionary
    results = {
        'Factor': [],
        'Method': [],
        'Model': [],
        'Best_f1score_macro': [],
        'Avg_F1Score_macro': [],
        'Std_f1score_macro': [],
        'Best_accuracy': [],
        'Avg_acc': [],
        'Std_acc': [],
        'All_accuracies': [],
        'All_f1_scores': []
    }

    # List to store accuracies and f1-scores for all runs
    all_accuracies = []
    all_f1_scores = []

    print("Fit model:")
    for i in range(num_repeats):
        print(f"Experiment {i+1}/{num_repeats}")

        # Fit the model
        model.fit(trainx, trainy, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, validation_data=(testX, testy))

        # Evaluate model
        print("Evaluate model: ")
        loss, accuracy = model.evaluate(testx, testy, verbose=verbose)
        testing_pred = model.predict(testx)
        testing_pred = testing_pred.argmax(axis=-1)
        true_labels = testy.argmax(axis=-1)
        average_fscore_macro = f1_score(
            true_labels, testing_pred, average="macro")

        # Append results for this run
        all_accuracies.append(accuracy)
        all_f1_scores.append(average_fscore_macro)

    # Compute aggregated results
    best_f1score_macro = max(all_f1_scores)
    avg_F1Score_macro = np.mean(all_f1_scores)
    std_f1score_macro = np.std(all_f1_scores)
    best_accuracy = max(all_accuracies)
    avg_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)

    # Save results for all runs
    results['Factor'].append(factor)
    results['Method'].append(method)
    results['Model'].append(modelname)
    results['Best_f1score_macro'].append(best_f1score_macro)
    results['Avg_F1Score_macro'].append(avg_F1Score_macro)
    results['Std_f1score_macro'].append(std_f1score_macro)
    results['Best_accuracy'].append(best_accuracy)
    results['Avg_acc'].append(avg_acc)
    results['Std_acc'].append(std_acc)
    results['All_accuracies'].append(','.join(map(str, all_accuracies)))
    results['All_f1_scores'].append(','.join(map(str, all_f1_scores)))

    # Save results
    save_results_to_csv(results)


testy = to_categorical(testy)

for method in methods:
    # looping over Families
    if method == 'None':
        label = to_categorical(trainy)
        n_outputs = label.shape[1]
        print('aug_label shape:{}, testy_shape:{}, n_outputs: {}'.format(
            label.shape, testy.shape, n_outputs))
        createandevaluate(trainx=trainX, testx=testX, trainy=label, testy=testy, n_timesteps=n_timesteps, n_features=n_features, n_outputs=n_outputs,
                          activationConv=activationConv, verbose=verbose, epochs=epochs, batch_size=batch_size, factor=0, method=method)
    else:
        for factor in factors:
            print('Evaluating models for {} and factor={}'.format(method, factor))
            aug_3d, aug_label = augmenter(trainX, trainy, factor, method, nr_samples)  # output x= 3d, y= 1d
            print('Shape of augmented data:{}, shape of augmented labels:{}'.format(
                aug_3d.shape, aug_label.shape))
            data = np.concatenate((trainX, aug_3d), axis=0)
            label = np.concatenate((trainy, aug_label), axis=0)
            print('Data shape of after concatenation:{},Label shape after concatenation:{}'.format(
                data.shape, label.shape))
            # Ensure one-hot encoding matches n_outputs
            label = to_categorical(label)

            n_outputs = label.shape[1]
            print('aug_label shape:{}, testy_shape:{}, n_outputs: {}'.format(
                label.shape, testy.shape, n_outputs))
            createandevaluate(trainx=data, testx=testX, trainy=label, testy=testy, n_timesteps=n_timesteps, n_features=n_features, n_outputs=n_outputs,
                              activationConv=activationConv, verbose=verbose, epochs=epochs, batch_size=batch_size, factor=factor, method=method)

            del aug_3d, aug_label, data, label
            tf.keras.backend.clear_session()
            gc.collect()
