# This data Augmentation function take the input as 3D array and give the output as a 3D array
# The input data should be the train data set only. 
# THe used python library (TSAUG) takes the X nad y data in 3D format.
#----> INPUT FORMATE x = (Samples, Row, Column) 3D, Y = labels (Samples, Row, Columns), 3D
#----> OUTPUT FORMATE x = (Samples, Row, Columns) 3D, Y = labels (Samples, Row, Column), 3D


import numpy as np
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve, Pool, AddNoise
import numpy as np
from DTW import *
import synthetic_data_generator

# -----------------Creating the augmenter class---------------------------------------------------
def augmenter(aug_method, factor):
    if aug_method == "time_warping":
        aug = (TimeWarp() * factor)  # random time warping 16 times in parallel)
    elif aug_method == "crop" or aug_method =="slicing":
        aug = (Crop(size= 128) * factor)    # random crop subsequences with length 16)
    elif aug_method == "jitter":
        aug = (AddNoise(loc=0.0, scale=0.1, distr='gaussian', kind='additive') * factor)  # Loc: Mean of the random noise. scale: The standard deviaton of the random noise. We camn use Nornmnal
    elif aug_method == "convolve" or aug_method =="magnitude_warping":
        aug = (Convolve(window="flattop", size=16) *factor)  # Convolve time series with a kernel window OF 16.
    elif aug_method == "rotation":
        aug = (Reverse() @ 0.8 *factor)  # with 50% probability, reverse the sequence
    elif aug_method == "quantize":
        aug = (Quantize(n_levels=[10, 20, 30]) *factor)  # random quantize to 10-, 20-, or 30- level sets
    elif aug_method == "drift" or aug_method =="scaling":
        aug = (Drift(max_drift=(0.1, 0.5)) @ 0.8 *factor)  # with 80% probability, random drift the signal up to 10% - 50%
    elif aug_method == "pool":
        aug = (Pool(size=10) *factor)  # Reduce the temporal resolution without changing the length       
    return aug

def augment(aug_factor, aug_method,x,y):
    aug_factor_type = type(aug_factor)
    if (aug_factor_type != int) and (aug_factor_type!= float):
            raise ValueError(f"Param 'aug_factor' should be numeric but received '{aug_factor}' with type '{aug_factor_type}'.")
    if aug_method == "crop" or aug_method == "time_warping" or aug_method == "jitter" or aug_method == "convolve" or aug_method == "rotation" or aug_method == "quantize" or aug_method == "drift" or aug_method == "pool":   
    # Convertig y to 3D. Becasue all the TSAUG family accpet the input (x and y) in 3D format. Here our input x is 3d and Y is 2 D
        L = 128 
        ov = 0
        def get_strides(a, L, ov):
            out = []
            for i in range(0, a.shape[0] - L + 1, L - ov):
                out.append(a[i:i + L, :])
            return np.array(out)
    
        y = np.expand_dims(y, axis=1) # Changing shape from 1D to 2D
        y = np.repeat(y, 128, axis=0)  # Converting the labels into total number of time stamp
        y = get_strides(y, L, ov)
        y = y.reshape(y.shape[0], y.shape[1], 1) # Changing shape from 2D to 3D
        
        if ((aug_factor>0) and (aug_factor<1)) :
            factor = 1
            aug = augmenter(aug_method, factor)
            x_aug, y_aug = aug.augment(x, y) # Both x_aug and y_aug are 3D
            mask = int(aug_factor* x.shape[0])
            print('mask is:', mask)
            x_aug = x_aug[:mask] # 3D
            y_aug = y_aug[:mask] # 3D
        elif aug_factor_type == int :
            factor = aug_factor
            aug = augmenter(aug_method, factor)
            x_aug, y_aug = aug.augment(x, y) # Both x and y are 3D
        
        else :
            print("The augmentation factor must be greater than 0 and less than and 2")

        #Converting y_aug to 1D array
        nb_segments = y_aug.shape[0]
        y_aug = y_aug.reshape(y_aug.shape[0], y_aug.shape[1]) # Converting to 2D array
        labels_to_save = np.zeros(nb_segments, dtype=int)
        for i in range(0, nb_segments):
            labels = y_aug[i][:]
            values, counts = np.unique(labels, return_counts=True)
            labels_to_save[i] = values[np.argmax(counts)]

        #return x_aug, y_aug
        return x_aug, labels_to_save 
    
    elif aug_method == "TW" or aug_method == "RGW" or aug_method == "DGW" or aug_method == "spawner" or aug_method == "permutation":
               
        if ((aug_factor >0) and (aug_factor<1)):
            mask = int(aug_factor * x.shape[0])
            x_frac = x[:mask]
            y_frac = y[:mask]
            x_aug, y_aug = globals()[aug_method](x_frac, y_frac) # x_aug is 3D and y_aug is  1D
        
        elif aug_factor_type == int:
            x_aug = []
            y_aug = []
            for i in range(aug_factor):
                x_aug_den, y_aug_den = globals()[aug_method](x, y)
                x_aug.append(x_aug_den)
                y_aug.append(y_aug_den)             
            x_aug, y_aug = np.vstack(x_aug), np.vstack(y_aug)
            y_aug = y_aug.flatten()
            #x_aug, y_aug = np.array(x_aug), np.array(y_aug)
    
    elif aug_method == 'cGAN':
            # Generate the  synthetic data from the Generator
            X_synthetic, Y_synthetic = synthetic_data_generator.GAN_generator()    
            x_aug = np.concatenate((x_aug, X_synthetic), axis=0)
            y_aug = np.concatenate((y_aug, Y_synthetic), axis= 0)
    else:
            print("The augmentation factor must be greater greater than 0")
    return x_aug, y_aug     
     
        
if __name__ == "__main__":
    print(__name__)
    try :
        x_aug, labels_to_save = augment(aug_factor, aug_method, x,y)
    except: 
        print("Some thing is wrong with the Augmente, Probably the augmentation factor, x, y and the aug_method is not declared.")

