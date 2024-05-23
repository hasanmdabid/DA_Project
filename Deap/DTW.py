import utils.augmentation as aug
import numpy as np

# All the Augmentation Function Here take input Data (Nr Samples, row, column- 3D) 
# and Labels (Nr of Sagments, 1D) 

def DGW(data, labels):   

    guided_DTW_1 = aug.discriminative_guided_warp(data, labels)
    return guided_DTW_1, labels

def RGW(data, labels):
    guided_RTW_1 = aug.random_guided_warp(data, labels)
    return guided_RTW_1, labels

def TW(data, labels):
    data_aug = aug.time_warp(data)
    return data_aug, labels

def permutation(data, labels):
    data_aug = aug.permutation(data)
    return data_aug, labels

def spawner(data, labels):
    data_aug = aug.spawner(data, labels)
    return data_aug, labels


def WW(data, labels):
    data_aug = aug.window_warp(data, window_ratio=0.1, scales=[0.5, 2.])
    return data_aug, labels