import utils.augmentation as aug

# All the Augmentation Function Here take input Data (Nr Samples, row, column- 3D) 
# and Labels (Nr of Sagments, 1D) 

def jitter(data, labels):
    data_aug = aug.jitter(data)
    return data_aug, labels

def scaling(data, labels):
    data_aug = aug.scaling(data)
    return data_aug, labels

def rotation(data, labels):
    data_aug = aug.rotation(data)
    return data_aug, labels


def magnitude_warp(data, labels):
    data_aug = aug.magnitude_warp(data)
    return data_aug, labels

def slicing(data, labels):
    data_aug = aug.slicing(data)
    return data_aug, labels

def TW(data, labels):
    data_aug = aug.time_warp(data)
    return data_aug, labels

def WW(data, labels):
    data_aug = aug.window_warp(data, window_ratio=0.1, scales=[0.5, 2.])
    return data_aug, labels

def permutation(data, labels):
    data_aug = aug.permutation(data)
    return data_aug, labels

def DGW(data, labels):   

    guided_DTW = aug.discriminative_guided_warp(data, labels)
    return guided_DTW, labels

def RGW(data, labels):
    guided_RTW = aug.random_guided_warp(data, labels)
    return guided_RTW, labels

def spawner(data, labels):
    data_aug = aug.spawner(data, labels)
    return data_aug, labels


