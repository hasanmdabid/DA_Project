# This data Augmentation function take the input ar 2D array and give the output as a 3D array
# The input data should be the train data set only. 
#----> INPUT fORMATE X = Samples, Row, Column
#----> output FORMATE Y = Samples, Row, Column(1)


import numpy as np
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve, Pool, AddNoise
import pandas as pd
import numpy as np


def seg_TSAUG(x, y, Aug_factor):
    L = 128
    ov = 0

    def get_strides(a, L, ov):
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])
        return np.array(out)

    # In this function the input x is 3d and y is a 1d array 
   
    y = np.expand_dims(y, axis=1)
    y = np.repeat(y, 128, axis=0)  # Converting the labels into total number of time stamp
    y = get_strides(y, L, ov)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    # -----------------Creating the augmenter class---------------------------------------------------
    #time_augmenter = (TimeWarp() * Aug_factor)  # random time warping 16 times in parallel)
    crop_augmenter = (Crop(size=L) * Aug_factor)  # random crop subsequences with length 16)
    #jittering_augmenter = (AddNoise(loc=0.0, scale=0.1, distr='gaussian', kind='additive') * Aug_factor)  # Loc: Mean of the random noise. scale: The standard deviaton of the random noise. We camn use Nornmnal
    #convolve_augmenter = (Convolve(window="flattop", size=16) * Aug_factor)  # Convolve time series with a kernel window OF 16.
    #rotation_augmenter = (Reverse() @ 0.8 * Aug_factor)  # with 50% probability, reverse the sequence
    #quantize_augmenter = (Quantize(n_levels=[10, 20, 30]) * Aug_factor)  # random quantize to 10-, 20-, or 30- level sets
    #drift_augmenter = (Drift(max_drift=(0.1, 0.5)) @ 0.8 * Aug_factor)  # with 80% probability, random drift the signal up to 10% - 50%
    #pool_augmenter = (Pool(size=10) * Aug_factor)  # Reduce the temporal resolution without changing the length

    x_1st_aug, y_1st_aug = crop_augmenter.augment(x, y)
    
    x = np.concatenate((x, x_1st_aug), axis=0)
    y = np.concatenate((y, y_1st_aug), axis=0)

    nb_segments = y.shape[0]
    y = y.reshape(y.shape[0], y.shape[1])
    labels_to_save = np.zeros(nb_segments, dtype=int)
    for i in range(0, nb_segments):
        labels = y[i][:]
        values, counts = np.unique(labels, return_counts=True)
        labels_to_save[i] = values[np.argmax(counts)]

    return x, labels_to_save