import load_data
import numpy as np
import pandas as pd
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve, Pool
from keras.utils import to_categorical
import math


def seg_TSAUG(data):
    """
    def column_notation(data):
        data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                        '18',
                        '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                        '35',
                        '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51',
                        '52',
                        '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68',
                        '69',
                        '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85',
                        '86',
                        '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101',
                        '102',
                        '103', '104', '105', '106', '107', 'Activity_Label']
        data['Activity_Label'] = data['Activity_Label'].replace(
            [406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511, 406508,
             404508,
             408512, 407521, 405506], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        return data

    data = column_notation(data)

    # Activities list
    activities = {
        'Old label': [0, 406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511,
                      406508, 404508, 408512, 407521, 405506],
        'New label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        'Activity': ['No activity', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'close Door 2', 'Open fridge',
                     'Close fridge',
                     'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2',
                     'Close Drawer 2',
                     'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup', 'Toggle Switch']}
    activities = pd.DataFrame(activities)
    # print(activities)

    """

    def get_strides(a, L, ov):
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])
        return np.array(out)

    # priority = input('With Majority?')
    priority = 'YES'
    if priority == 'YES':
        ##========================Seperatiing the Null Class from the data to generate Only the Minority class============
        data_maj = data.loc[data.Activity_Label == 0]
        # print('Shape of DATA with Majority target', data_maj.shape)

        data_min = data.loc[data.Activity_Label > 0]
        # print('Shape of DATA with Minority target', data_min.shape)

        data_maj = data_maj.to_numpy()
        data_maj = get_strides(data_maj, 32, 16)
        # print('Shape of DATA with Majority target after slided window', data_maj.shape)
        data_min = data_min.to_numpy()
        data_min = get_strides(data_min, 32, 16)
        # print('Shape of DATA with Minority target after slided window', data_min.shape)
        # print(data.shape)

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
        # print('Factor:', factor)

        my_augmenter = (
                TimeWarp() * factor  # random time warping 16 times in parallel
            #    Crop(size=32) * 8  # # random crop subsequences with length 16
            #        + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
            #    Drift(max_drift=(0.1, 0.5)) @ 0.8 * 8  # with 80% probability, random drift the signal up to 10% - 50%
            #        Reverse() @ 0.8 * 8  # with 50% probability, reverse the sequence
            #    Convolve(window="flattop", size=16) * 8 # Convolve time series with a kernel window OF 16.
            #     Pool(size=10) * 8  # Reduce the temporal resolution without changing the length.
        )

        X_aug, Y_aug = my_augmenter.augment(x_min, y_min)

        """""

        print('After Augmentation of Min clas only:')
        print(X_aug.shape)
        print(Y_aug.shape)



        unique_aug, counts_aug = np.unique(Y_aug, return_counts=True)


        print('After Augmentation: Classes and Instances')
        print(np.asarray((unique_aug, counts_aug)).T) 
        """
        x = np.concatenate((x_maj, X_aug), axis=0)
        y = np.concatenate((y_maj, Y_aug), axis=0)

        """"
        print('After Concate x:', x.shape)
        print('After concate y:', y.shape)
        """
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

        """
        print('Before Augmentation:')
        print('Shape of X:', x.shape)
        print('Shape of y:', y.shape)
        unique, counts = np.unique(y, return_counts=True)
        print('Before Augmentation: Classes and Instances')
        print(np.asarray((unique, counts)).T)
        """

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

        """
        print('After Augmentation:')
        print(X_aug.shape)
        print(Y_aug.shape)

        unique_aug, counts_aug = np.unique(Y_aug, return_counts=True)

        print('After Augmentation: Classes and Instances')
        print(np.asarray((unique_aug, counts_aug)).T)
        """
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


