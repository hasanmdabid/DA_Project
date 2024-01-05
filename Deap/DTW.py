import utils.augmentation as aug
import numpy as np
import numpy as np
import utils.augmentation as aug

# All the Augmentation Function Here take input Data (Nr Samples, row, column- 3D) 
# and Labels (Nr of Sagments, 1D) 

def DGW(data, labels):   

    guided_DTW_1 = aug.discriminative_guided_warp(data, labels)
    return guided_DTW_1, labels

def RGW(data, labels):
    guided_RTW_1 = aug.random_guided_warp(data, labels)
    return guided_RTW_1, labels

def TW(data, labels):
    guided_TW_1 = aug.time_warp(data)
    return guided_TW_1, labels

def permutation(data, labels):
    data_aug = aug.permutation(data)
    return data_aug, labels

def spawner(data, labels):
    data_aug = aug.spawner(data, labels)
    return data_aug, labels
