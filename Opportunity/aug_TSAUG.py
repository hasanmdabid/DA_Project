# This data Augmentation function take the input as 3D array and give the output as a 3D array
# The input data should be the train data set only. 
#----> INPUT FORMATE x = (Samples, Row, Column) 3D, Y = labels (1D)
#----> OUT FORMATE Y = Samples, Row, Column(1)

# pylint: disable-all
import load_data
import numpy as np
import pandas as pd
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve, Pool, AddNoise
from keras.utils import to_categorical
import math


def TSAUG(data):
    L = 32
    ov = 0

    def get_strides(a, L, ov):
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])
        return np.array(out)

    # priority = input('Without Majority?')

    priority = 'YES'
    if priority == 'YES':
        ##========================Seperatiing the Null Class from the data to generate Only the Minority class============
        data_maj = data.loc[data.Activity_Label == 0]
        # print('Shape of DATA with Majority target', data_maj.shape)

        data_min = data.loc[data.Activity_Label > 0]
        # print('Shape of DATA with Minority target', data_min.shape)

        data_maj = data_maj.to_numpy()
        data_maj = get_strides(data_maj, L, ov)
        # print('Shape of DATA with Majority target after slided window', data_maj.shape)
        data_min = data_min.to_numpy()
        data_min = get_strides(data_min, L, ov)
        # print('Shape of DATA with Minority target after slided window', data_min.shape)


        x_maj = data_maj[:, :, :-1].astype('float32')
        # print('Majority Feature class shape:', x_maj.shape)

        y_maj = data_maj[:, :, -1].astype('int')
        # print('Majority Target class shape:', y_maj.shape)
        y_maj = y_maj.reshape(y_maj.shape[0], y_maj.shape[1], 1)
        # print('Majority Target class after reshape:', y_maj.shape)

        x_min = data_min[:, :, :-1].astype('float32')
        y_min = data_min[:, :, -1].astype('int')
        y_min = y_min.reshape(y_min.shape[0], y_min.shape[1], 1)

        factor = math.floor(y_maj.shape[0] * y_maj.shape[1] * 0.001)
        #factor = 1                                         #   ************** Augmentation factor ****************** 
        # Here Factor value decides the synthetic data volume. Instead of selecting a random value we propose the following formula,
        # factor = Nr of Majority class Instances/1000
        print('Factor:', factor)

        # -----------------Creating the augmenter class---------------------------------------------------
        time_augmenter = (TimeWarp() * factor)  # random time warping Factor times in parallel)
        crop_augmenter = (Crop(size=L) * factor)  # random crop subsequences with length 16)
        jittering_augmenter = (AddNoise(loc=0.0, scale=0.1, distr='gaussian', kind='additive') * factor)  # Loc: Mean of the random noise. scale: The standard deviaton of the random noise. We camn use Nornmnal
        convolve_augmenter = (Convolve(window="flattop", size=16) * factor)  # Convolve time series with a kernel window OF 16.
        rotation_augmenter = (Reverse() @ 0.8 * factor)  # with 50% probability, reverse the sequence
        quantize_augmenter = (Quantize(n_levels=[10, 20, 30]) * factor)  # random quantize to 10-, 20-, or 30- level sets
        drift_augmenter = (Drift(max_drift=(0.1, 0.5)) @ 0.8 * factor)  # with 80% probability, random drift the signal up to 10% - 50%
        pool_augmenter = (Pool(size=10) * factor)  # Reduce the temporal resolution without changing the length

        x_aug, y_aug = time_augmenter.augment(x_min, y_min)
        #x_aug, y_aug = jittering_augmenter.augment(x_min, y_min)
        #x_aug, y_aug = convolve_augmenter.augment(x_min, y_min)
        #x_aug, y_aug = rotation_augmenter.augment(x_min, y_min)
        #x_aug, y_aug = quantize_augmenter.augment(x_min, y_min)
        #x_aug, y_aug = drift_augmenter.augment(x_min, y_min)
        #x_aug, y_aug = pool_augmenter.augment(x_min, y_min)


        x = np.concatenate((x_maj, x_min, x_aug), axis=0)
        y = np.concatenate((y_maj, y_min, y_aug), axis=0)

        nb_segments = y.shape[0]
        y = y.reshape(y.shape[0], y.shape[1])
        labels_to_save = np.zeros(nb_segments, dtype=int)
        for i in range(0, nb_segments):
            labels = y[i][:]
            values, counts = np.unique(labels, return_counts=True)
            labels_to_save[i] = values[np.argmax(counts)]

        return x, labels_to_save

    elif priority == 'NO':
        data = data.to_numpy()
        data = get_strides(data, 32, 16)
        # print(data.shape)

        x = data[:, :, :-1].astype('float32')
        y = data[:, :, -1].astype('int')
        y = y.reshape(y.shape[0], y.shape[1], 1)

        print('Before Augmentation:')
        print('Shape of X:', x.shape)
        print('Shape of y:', y.shape)
        unique, counts = np.unique(y, return_counts=True)
        print('Before Augmentation: Classes and Instances')
        print(np.asarray((unique, counts)).T)

        my_augmenter = (
                TimeWarp() * 5  # random time warping 16 times in parallel
            #   Crop(size=32) * 8  # # random crop subsequences with length 16
            #        + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
            #   Drift(max_drift=(0.1, 0.5)) @ 0.8 *5 # with 80% probability, random drift the signal up to 10% - 50%
            #   Reverse() @ 0.8 * 5  # with 80% probability, reverse the sequence
            #   Convolve(window="flattop", size=8) *5 # Convolve time series with a kernel window OF 8.
            #   Pool(size=10) *5  # Reduce the temporal resolution without changing the length.
        )

        X_aug, Y_aug = my_augmenter.augment(x, y)

        print('After Augmentation:')
        print(X_aug.shape)
        print(Y_aug.shape)

        unique_aug, counts_aug = np.unique(Y_aug, return_counts=True)

        print('After Augmentation: Classes and Instances')
        print(np.asarray((unique_aug, counts_aug)).T)

        nb_segments = Y_aug.shape[0]
        Y_aug = Y_aug.reshape(Y_aug.shape[0], Y_aug.shape[1])
        labels_to_save = np.zeros(nb_segments, dtype=int)
        for i in range(0, nb_segments):
            labels = Y_aug[i][:]
            values, counts = np.unique(labels, return_counts=True)
            labels_to_save[i] = values[np.argmax(counts)]

        # print(labels_to_save.shape)
        return X_aug, labels_to_save


# To check the Function <---------------------------------------------------------------------
"""
s1r1 = load_data.load_data("S1-ADL1")
trainxs1r1, trainys1r1 = seg_TSAUG(s1r1)
print(trainys1r1.shape)
unique, counts = np.unique(trainys1r1, return_counts=True)
print(np.asarray((unique, counts)).T)
trainys1r1 = to_categorical(trainys1r1)
print(trainxs1r1.shape)
print(trainys1r1.shape)
"""
