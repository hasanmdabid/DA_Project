# This file enlisted aLL the models from the previous researcher.
import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D, Conv2D

from keras.layers.convolutional import MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.layers import LSTM, Lambda, Reshape
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold

# Diclaring the Global variable.

global n_timesteps, n_features

def conv1D(activation='relu', init_mode='uniform', optimizer='rmsprop', dropout_rate=0.6):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation=activation, kernel_initializer=init_mode, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=64, kernel_size=3, activation=activation, kernel_initializer=init_mode ))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))

    model.add(Conv1D(filters=32, kernel_size=3, activation=activation, kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))

    model.add(Conv1D(filters=32, kernel_size=3, activation=activation, kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(32, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # Compiling The model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def conv2D(activation='relu', init_mode='glorot_uniform', optimizer='Adam', dropout_rate=0.6):
    model = Sequential()
    # Adding Batch normalization before CONV
    # model.add(BatchNormalization(input_shape=(n_timesteps, n_features, 1)))

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


def deepconvlstm(activation='relu', init_mode='glorot_uniform', optimizer='Adam', dropout_rate=0.2):
    # First adding the Batch Normalization layer.
    def ReshapeLayer(x):
        shape = x.shape

        # 1 possibility: H,W*channel
        reshape = Reshape((shape[1], shape[2] * shape[3]))(x)

        # 2 possibility: W,H*channel
        # transpose = Permute((2,1,3))(x)
        # reshape = Reshape((shape[1],shape[2]*shape[3]))(transpose)

        return reshape

    model = Sequential()
    # model.add(BatchNormalization(input_shape=(n_timesteps, n_features, 1)))
    # 1st convolutional + pooling
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activation, kernel_initializer=init_mode, input_shape=(n_timesteps, n_features, 1)))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    # 2nd Convolution
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activation, kernel_initializer=init_mode))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    # 3rd Convolution
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activation, kernel_initializer=init_mode))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    # 4th convolution
    model.add(Conv2D(filters=32, kernel_size=(3, 1), activation=activation, kernel_initializer=init_mode))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(dropout_rate))

    model.add(Lambda(ReshapeLayer))  # <========== pass from 4D to 3D
    # 1st LSTM layers
    model.add(LSTM(64, activation='tanh', return_sequences=True, kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))

    # 2nd LSTM layers
    model.add(LSTM(32, activation='tanh', return_sequences=False, kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(32, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # Compiling The model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def lstm(dropout_rate=0.2):
    model=Sequential()
    model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(32, activation='tanh', return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten())  
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])
    
    return model