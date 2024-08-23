# pylint: disable-all
import numpy as np
import utils.augmentation as aug
from utils.augmentation import *

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


def augmenter(data, factor, family):
    ##========================Seperatiing the Null Class from the data to generate Only the Minority class============
    data_maj = data.loc[data.Activity_Label == 0]
    #print('Shape of DATA with Majority target', data_maj.shape)
    
    data_maj = data_maj.to_numpy()

    data_min = data.loc[data.Activity_Label > 0]
    #print('Shape of DATA with Minority target', data_min.shape)
    data_min = data_min.to_numpy()
    
    data_Maj_3D, labels_maj_to_save = slided_numpy_array(data_maj)
    data_min_3D, labels_min_to_save = slided_numpy_array(data_min)

    aug_data = []  # Initialize aug_data to an empty list to avoid UnboundLocalError

    if family in ['jitter', 'scaling', 'rotation', 'permutation', 'magnitude_warp', 'slicing', 'time_warp', 'window_warp']:
        if 0 < factor < 1:
            mask = int(factor * data_min_3D.shape[0])
            x_frac = data_min_3D[:mask]
            repeated_labels_min = labels_min_to_save[:mask]
            aug_data = globals()[family](x_frac)  # x_aug is 3D and y_aug is 1D
        elif isinstance(factor, int):
            for i in range(factor):
                print(f"Running {family} {i+1}/{factor}")
                warped_data = globals()[family](x=data_min_3D)
                aug_data.append(warped_data)
            # Repeat the labels_min_to_save array 'factor' times
            repeated_labels_min = np.tile(labels_min_to_save, factor)
            # Concatenate the augmented data
            aug_data = np.concatenate(aug_data, axis=0)
        else:
            raise ValueError("Factor should be either an int or a float between 0 and 1")
    elif family in ['spawner', 'random_guided_warp', 'discriminative_guided_warp']:
        if 0 < factor < 1:
            mask = int(factor * data_min_3D.shape[0])
            x_frac = data_min_3D[:mask]
            repeated_labels_min = labels_min_to_save[:mask]
            aug_data = globals()[family](x=x_frac, labels=repeated_labels_min)  # x_frac is 3D and y_aug is 1D
        elif isinstance(factor, int):
            for i in range(factor):
                print(f"Running {family} {i+1}/{factor}")
                warped_data = globals()[family](x=data_min_3D, labels=labels_min_to_save)
                aug_data.append(warped_data)
            # Repeat the labels_min_to_save array 'factor' times
            repeated_labels_min = np.tile(labels_min_to_save, factor)
            # Concatenate the augmented data
            aug_data = np.concatenate(aug_data, axis=0)
        else:
            raise ValueError("Factor should be either an int or a float between 0 and 1")
    else:
        raise ValueError(f"Unknown family: {family}")

    # Concatenate the data arrays
    data_hybrid = np.concatenate((data_Maj_3D, data_min_3D, aug_data), axis=0)
    
    # Concatenate the labels arrays
    labels_hybrid = np.concatenate((labels_maj_to_save, labels_min_to_save, repeated_labels_min), axis=0)

    return data_hybrid, labels_hybrid


