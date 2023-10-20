# This file enlisted aLL the models from the previous researcher.
import os.path
from pathlib import Path
from datetime import datetime
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import ConvLSTM2D, ConvLSTM1D
from keras.layers import MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.layers import LSTM, Lambda, Reshape


def NORMCONV1D(n_timesteps, n_features, n_outputs, nkerns, filterSizes, poolSizes, activationConv, inputMLP,activationMLP, withBatchNormalization=True):
    model = Sequential()
    if withBatchNormalization:
        model.add(BatchNormalization(input_shape=(n_timesteps, n_features)))

    model.add(Conv1D(filters=64, kernel_size=3, activation=activationConv, input_shape=(n_timesteps, n_features)))

    model.add(Conv1D(filters=64, kernel_size=3, activation=activationConv))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))

    model.add(Conv1D(filters=32, kernel_size=3, activation=activationConv))

    model.add(Conv1D(filters=32, kernel_size=3, activation=activationConv))
    # Adding the Drop out layer at the last CONV1D layer to prevent overfillting.
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))

    model.add(Flatten())
    model.add(Dense(inputMLP[0], activation=activationMLP))
    model.add(Dense(n_outputs, activation='softmax'))

    return model


def CONV2D(n_timesteps, n_features, n_outputs, nkerns, filterSizes, activationConv, inputMLP, activationMLP, withBatchNormalization=True):
    model = Sequential()
    # Adding Batch normalization before CONV
    if withBatchNormalization:
        model.add(BatchNormalization(input_shape=(n_timesteps, n_features, 1)))

    # 1st convolutional + pooling
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activationConv,
                     input_shape=(n_timesteps, n_features, 1)))

    # 2nd convolutional + pooling + normalization layer
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activationConv))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))

    # 3rd block: convolutional + RELU + normalization
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activationConv))
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activationConv))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))

    # Fully-connected layer
    model.add(Flatten())
    model.add(Dense(inputMLP[0], activation=activationMLP))

    # Softmax layer
    model.add(Dense(n_outputs, activation='softmax'))

    return model


def DEEPCONVLSTM(n_timesteps, n_features, n_outputs, nkerns, filterSizes, poolSizes, activationConv, inputMLP, activationMLP):
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
    model.add(BatchNormalization(input_shape=(n_timesteps, n_features, 1)))
    # 1st convolutional + pooling
    model.add(
        Conv2D(filters=64, kernel_size=(3, 1), activation=activationConv, input_shape=(n_timesteps, n_features, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activationConv))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))

    # 2nd convolutional + pooling
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activationConv))
    model.add(Conv2D(filters=64, kernel_size=(3, 1), activation=activationConv))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding="valid"))
    model.add(Dropout(0.5))

    model.add(Lambda(ReshapeLayer))  # <========== pass from 4D to 3D
    # LSTM layers
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=False))

    model.add(Dense(512, activation=activationMLP))
    model.add(Dense(n_outputs, activation='softmax'))

    return model


def CONVLSTM_1D_NONHYBRID(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, inputMLP):
    model = Sequential()
    model.add(ConvLSTM1D(filters=filterSizes[0], kernel_size=nkerns[0][0], padding='same', activation='relu',
                         return_sequences=True, input_shape=(n_steps, n_length, n_features)))
    model.add(BatchNormalization())

    model.add(ConvLSTM1D(filters=filterSizes[0], kernel_size=nkerns[0][0], padding='same', activation='relu',
                         return_sequences=True))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(n_outputs, activation='Softmax'))
    return model


def CONVLSTM_1D(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, inputMLP, activationMLP, activationConv):
    model = Sequential()
    model.add(ConvLSTM1D(filters=filterSizes[0], kernel_size=nkerns[0][0], padding="same", return_sequences=True,
                         input_shape=(n_steps, n_length, n_features)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation=activationConv, padding='same'))
    model.add(AveragePooling2D(pool_size=(2, 1)))

    model.add(Conv2D(filters=filterSizes[0], kernel_size=(3, 3), activation=activationConv, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Flatten())

    model.add(Dense(256, activation=activationMLP))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation=activationMLP))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=activationMLP))
    model.add(Dense(n_outputs, activation='softmax'))

    return model


def CONVLSTM_2D_NONHYBRID(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, activationConv, activationMLP,                         
                          inputMLP):
    model = Sequential()
    model.add(ConvLSTM2D(filters=filterSizes[0], kernel_size=nkerns[0], padding='same', activation=activationConv,
                         return_sequences=True, input_shape=(n_steps, 1, n_length, n_features)))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=filterSizes[0], kernel_size=nkerns[0], padding='same', activation=activationConv,
                         return_sequences=True))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation=activationMLP))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(n_outputs, activation='Softmax'))
    return model


def CONVLSTM_2D(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, activationConv, activationMLP, inputMLP):
    model = Sequential()
    model.add(ConvLSTM2D(filters=filterSizes[0], kernel_size=nkerns[0], padding='same', activation=activationConv,
                         return_sequences=True, input_shape=(n_steps, 1, n_length, n_features)))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=filterSizes[2], kernel_size=(3, 3, 3), activation=activationConv, padding='same',
                     data_format='channels_last'))

    model.add(Flatten())
    model.add(Dense(128, activation=activationMLP))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=activationMLP))
    model.add(Dense(n_outputs, activation='Softmax'))
    return model


def saveResultsCSV(method, modelname, epochs, batch_size, accuracy, average_fscore_macro, average_fscore_weighted,
                   nKerns, filterSizes):
    path = './results/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = './results/results.csv'
    file = Path(fileString)

    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write(
            'Finished on; Modelname; Epochs; Batch_Size; accuracy; Average_fscore_Macro; Average_fscores_weighted; nKerns; filterSize\n')
        f.close()
    with open(fileString, "a") as f:
        f.write(
            '{};{};{};{};{};{};{};{};{}:{}\n'.format(now, method, modelname, epochs, batch_size, accuracy, average_fscore_macro,
                                                  average_fscore_weighted, nKerns, filterSizes))
    f.close()
