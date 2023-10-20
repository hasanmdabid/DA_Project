import scripts.augmentation as aug


# All the Augmentation Function Here take input Data (Nr Samples, row, column- 3D) 
# and Labels (Nr of Sagments, 1D) 

#***************To use the DTW algorithm the shape of the input is 
#***************x = 3D(Nr. segents, row, column) 
#***************y = 1D(Nr segments or labels)

#***************To use the DTW algorithm the shape of the output is 
#8**************x = 3D(Nr. segents, row, column) 
#***************y = 1D(Nr segments or labels)

def DGW(data, labels):   

    guided_DTW_1 = aug.discriminative_guided_warp(data, labels)
    return guided_DTW_1, labels

def RTW(data, labels):
    guided_RTW_1 = aug.random_guided_warp(data, labels)
    return guided_RTW_1, labels

def TW(data, labels):
    guided_TW_1 = aug.time_warp(data)
    return guided_TW_1, labels