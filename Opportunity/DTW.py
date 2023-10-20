
import numpy as np
import utils.augmentation as aug

# All the Function takes 3D input and outputs arrays are also 3D

#***************To use the TSAUG algorithm the shape of the input is x = 3D(Nr. segents, row, column) 
#***************y = 1D(Nr segments or labels)

#***************To use the TSAUG algorithm the shape of the output is x = 3D(Nr. segents, row, column) 
#***************y = 1D(Nr segments or labels)


def slided_numpy_array(data):

    import numpy as np
    x = data

    # This function will generate the slided Windowed Data

    def get_strides(a, L, ov):
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])

        return np.array(out)

    L = 32
    ov = 16

    # print('After Overlapping')
    x = get_strides(x, L, ov)
    # print(x.shape)

    segment_idx = 0  # Index for the segment dimension
    nb_segments, nb_timestamps, nb_columns = x.shape
    data_to_save = np.zeros((nb_segments, nb_timestamps, nb_columns - 1), dtype=np.float32)
    labels_to_save = np.zeros(nb_segments, dtype=int)

    for i in range(0, nb_segments):
        labels = x[i][:][:]
        data_to_save[i] = labels[:, :-1]
        labels = x[i][:][:]
        labels = labels[:, -1]
        labels = labels.astype('int')  # Convert labels to int to avoid typing issues
        values, counts = np.unique(labels, return_counts=True)
        labels_to_save[i] = values[np.argmax(counts)]

    return data_to_save, labels_to_save


def DTW(data):
    ##========================Seperatiing the Null Class from the data to generate Only the Minority class============
    data_maj = data.loc[data.Activity_Label == 0]
    #print('Shape of DATA with Majority target', data_maj.shape)
    
    data_maj = data_maj.to_numpy()

    data_min = data.loc[data.Activity_Label > 0]
    #print('Shape of DATA with Minority target', data_min.shape)
    data_min = data_min.to_numpy()
    
    data_Maj_3D, labels_maj_to_save = slided_numpy_array(data_maj)
    data_min_3D, labels_min_to_save = slided_numpy_array(data_min) 
    guided_DTW_1 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_2 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_3 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_4 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_5 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_6 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_7 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_8 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_9 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    #guided_DTW_10 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    
    data_hybrid = np.concatenate((data_Maj_3D, data_min_3D, guided_DTW_1, guided_DTW_2, guided_DTW_3, guided_DTW_4, guided_DTW_5, guided_DTW_6, guided_DTW_7, guided_DTW_8, guided_DTW_9), axis=0) 
    #labels_hybrid = np.concatenate((labels_maj_to_save.reshape(labels_maj_to_save.shape[0], 1), labels_min_to_save.reshape(labels_min_to_save.shape[0], 1),labels_min_to_save.reshape(labels_min_to_save.shape[0], 1)), axis=0)
    labels_hybrid = np.concatenate((labels_maj_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save,
                                    labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save), axis=0)
    return data_hybrid, labels_hybrid

def RTW(data):
    ##========================Seperatiing the Null Class from the data to generate Only the Minority class============
    data_maj = data.loc[data.Activity_Label == 0]
    #print('Shape of DATA with Majority target', data_maj.shape)
    
    data_maj = data_maj.to_numpy()

    data_min = data.loc[data.Activity_Label > 0]
    #print('Shape of DATA with Minority target', data_min.shape)
    data_min = data_min.to_numpy()
    
    data_Maj_3D, labels_maj_to_save = slided_numpy_array(data_maj)
    data_min_3D, labels_min_to_save = slided_numpy_array(data_min) 
    guided_DTW_1 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_2 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_3 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_4 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_5 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_6 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_7 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_8 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    guided_DTW_9 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    #guided_DTW_10 = aug.random_guided_warp(data_min_3D, labels_min_to_save)
    
    data_hybrid = np.concatenate((data_Maj_3D, data_min_3D, guided_DTW_1, guided_DTW_2, guided_DTW_3, guided_DTW_4, guided_DTW_5, guided_DTW_6, guided_DTW_7, guided_DTW_8, guided_DTW_9), axis=0) 
    #labels_hybrid = np.concatenate((labels_maj_to_save.reshape(labels_maj_to_save.shape[0], 1), labels_min_to_save.reshape(labels_min_to_save.shape[0], 1),labels_min_to_save.reshape(labels_min_to_save.shape[0], 1)), axis=0)
    labels_hybrid = np.concatenate((labels_maj_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save,
                                    labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save), axis=0)
    return data_hybrid, labels_hybrid

def TW(data):
    ##========================Seperatiing the Null Class from the data to generate Only the Minority class============
    data_maj = data.loc[data.Activity_Label == 0]
    #print('Shape of DATA with Majority target', data_maj.shape)
    
    data_maj = data_maj.to_numpy()

    data_min = data.loc[data.Activity_Label > 0]
    #print('Shape of DATA with Minority target', data_min.shape)
    data_min = data_min.to_numpy()
    
    data_Maj_3D, labels_maj_to_save = slided_numpy_array(data_maj)
    data_min_3D, labels_min_to_save = slided_numpy_array(data_min) 
    guided_DTW_1 = aug.time_warp(data_min_3D)
    guided_DTW_2 = aug.time_warp(data_min_3D)
    guided_DTW_3 = aug.time_warp(data_min_3D)
    guided_DTW_4 = aug.time_warp(data_min_3D)
    guided_DTW_5 = aug.time_warp(data_min_3D)
    guided_DTW_6 = aug.time_warp(data_min_3D)
    guided_DTW_7 = aug.time_warp(data_min_3D)
    guided_DTW_8 = aug.time_warp(data_min_3D)
    guided_DTW_9 = aug.time_warp(data_min_3D)
    #guided_DTW_10 = aug.time_warp(data_min_3D)
    
    data_hybrid = np.concatenate((data_Maj_3D, data_min_3D, guided_DTW_1, guided_DTW_2, guided_DTW_3, guided_DTW_4, guided_DTW_5, guided_DTW_6, guided_DTW_7, guided_DTW_8, guided_DTW_9), axis=0) 
    #labels_hybrid = np.concatenate((labels_maj_to_save.reshape(labels_maj_to_save.shape[0], 1), labels_min_to_save.reshape(labels_min_to_save.shape[0], 1),labels_min_to_save.reshape(labels_min_to_save.shape[0], 1)), axis=0)
    labels_hybrid = np.concatenate((labels_maj_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save,
                                    labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save), axis=0)
    return data_hybrid, labels_hybrid



