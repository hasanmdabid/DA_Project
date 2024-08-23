# pylint: disable-all
import numpy as np
from augmentation import *
from convert3D import slided_numpy_array
from tsaug_aug import*
import synthetic_data_generator
# ***************To use the TSAUG algorithm the shape of the input is x = 3D(Nr. segents, row, column)
# ***************y = 1D(Nr segments or labels)



def augmenter(data_3d, labels, factor, method, nr_samples):
    
    aug_data = []  # Initialize aug_data to an empty list to avoid UnboundLocalError
    
    def get_strides(a, L, ov):
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])
        return np.array(out)


    #data_3d, labels = slided_numpy_array(data, l, ov)
    if method in ["convolve", "quantize", "drift", "pool"]:
        # Convertig y to 3D. Becasue all the TSAUG family accpet the input (x and y) in 3D format. Here our input x is 3d and Y is 2 D
        L = 128
        ov = 0

        labels = np.expand_dims(labels, axis=1)  # Changing shape from 1D to 2D
        labels = np.repeat(labels, 128, axis=0)
        labels = get_strides(labels, L, ov)
        # Changing shape from 2D to 3D
        labels = labels.reshape(labels.shape[0], labels.shape[1], 1)

        if ((factor > 0) and (factor < 1)):
            dummy_factor = 1   # Firt we will augment the data in to 1 factor. then we will chunk the augmented data by multiplying the aug method
            aug = tsaug(method, dummy_factor)
            aug_data, labels_3d = aug.augment(data_3d, labels)  # Both x_aug and y_aug are 3D
            mask = int(factor * data_3d.shape[0])
            print('mask is:', mask)
            aug_data = aug_data[:mask]  # 3D
            labels_3d = labels_3d[:mask]  # 3D
        elif factor >= 1 :
            aug = tsaug(method, factor)
            aug_data, labels_3d = aug.augment(data_3d, labels)  # Both x and labels are 3D
        else :
            print("The augmentation factor must be greater than 0")

        #Converting y_aug to 1D array
        nb_segments = labels_3d.shape[0]
        labels_2d = labels_3d.reshape(labels_3d.shape[0], labels_3d.shape[1]) # Converting to 2D array
        aug_label = np.zeros(nb_segments, dtype=int)
        for i in range(0, nb_segments):
            labels = labels_2d[i][:]
            values, counts = np.unique(labels, return_counts=True)
            aug_label[i] = values[np.argmax(counts)]

        return aug_data, aug_label

    elif method in ['jitter', 'scaling', 'rotation', 'permutation', 'magnitude_warp', 'slicing', 'time_warp', 'window_warp']:
        if 0 < factor < 1:
            mask = int(factor * data_3d.shape[0])
            x_frac = data_3d[:mask]
            aug_label = labels[:mask]
            print(x_frac.shape, aug_label.shape)
            aug_data = globals()[method](x=x_frac)  # x_aug is 3D and y_aug is 1D
        elif isinstance(factor, int):
            for i in range(factor):
                print(f"Running {method} {i+1}/{factor}")
                warped_data = globals()[method](x=data_3d)
                aug_data.append(warped_data)
            # Repeat the labels array 'factor' times
            aug_label = np.tile(labels, factor)
            # Concatenate the augmented data
            aug_data = np.concatenate(aug_data, axis=0)
        else:
            raise ValueError(
                "Factor should be either an int or a float between 0 and 1")
    elif method in ['spawner','random_guided_warp','discriminative_guided_warp']:
        if 0 < factor < 1:
            mask = int(factor * data_3d.shape[0])
            x_frac = data_3d[:mask]
            aug_label = labels[:mask]
            aug_data = globals()[method](x=x_frac, labels=aug_label)  # x_frac is 3D and y_aug is 1D
        elif isinstance(factor, int):
            for i in range(factor):
                print(f"Running {method} {i+1}/{factor}")
                warped_data = globals()[method](
                    x=data_3d, labels=labels)
                aug_data.append(warped_data)
            # Repeat the labels array 'factor' times
            aug_label = np.tile(labels, factor)
            # Concatenate the augmented data
            aug_data = np.concatenate(aug_data, axis=0)
        else:
            raise ValueError(
                "Factor should be either an int or a float between 0 and 1")
    
    elif method == 'cGAN':
        # Generate the  synthetic data from the Generator
        aug_data, aug_label = synthetic_data_generator.GAN_generator(factor, nr_samples)
                      
    else:
        raise ValueError(f"Unknown method: {method}")
    """
    # Concatenate the data arrays
    data_hybrid = np.concatenate((data_3d, aug_data), axis=0)

    # Concatenate the labels arrays
    labels_hybrid = np.concatenate((labels, aug_label), axis=0)

    return data_hybrid, labels_hybrid
    """
    
    return aug_data, aug_label


print(__name__)
if __name__ == '__main__':
    try:
        aug_data, aug_label = augmenter(data, factor, method, l, ov)
    except TypeError:
        print("Something went wrong")
    except NameError:
        print("Data, factor, l, ov are not Defined")
