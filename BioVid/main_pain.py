import os
from matplotlib import axis
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 - all logs shown
# 1 - filter out INFO logs
# 2 - additionally filter out WARNING logs
# 3 - additionally filter out ERROR logs

import platform
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.stats import zscore
from scripts.augmentation import augment
from sklearn.utils import resample
from tensorflow.python.keras.utils.np_utils import to_categorical
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve, Pool, AddNoise

from hcf import get_hcf, moving_average
from scripts.classifier import *
from DTW import *
from scripts.preprocessing import remove_ecg_wandering, preprocess_np
from scripts.evaluation import loso_cross_validation, five_loso, accuracy, from_categorical
from scripts.data_handling import read_biovid_np, pick_classes, normalize, resample_axis, read_painmonit_np, normalize_features

from config import painmonit_sensors, biovid_sensors, sampling_rate_biovid, sampling_rate_painmonit

#-------------------------------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------------------------------


def prepare_data(X, y, subjects, param):

    x_aug, y_aug, subjects_aug = None, None, None

    # remove ECG wandering in BioVid dataset
    if "ecg" in param["selected_sensors"]:
        ecg_index = param["selected_sensors"].index("ecg")
        X[:, :, ecg_index, :] = np.apply_along_axis(func1d= remove_ecg_wandering, axis= 1, arr=X[:, :, ecg_index, :])

    if "preprocess" in param and param["preprocess"]:
        print("Preprocess signals...")
        X = preprocess_np(X, sensor_names= param["sensor_names"], sampling_rate= param["resample"])
        print("Signals preprocessed.")

    # --------------------------------------- HCF
    hcf = get_hcf(dataset= param["dataset"])

    # select sensors
    column_names = hcf.columns.values
    # All column names that start with sensor strings in "selected sensors"
    sensor_columns = [x for x in column_names for name in param["selected_sensors"] if x.startswith(name)]
    # Select columns
    hcf = hcf[sensor_columns]

    if "hcf_norm" in param and param["hcf_norm"]:
        hcf = normalize_features(hcf)

    hcf = hcf.fillna(0)

    # ------------------------------------------ Raw
    if "cut" in param and param["cut"] is not None:
        start = int(param["input_fs"] * param["cut"][0])
        end = int(param["input_fs"] * param["cut"][1])
        X = X[:, start:end]

    if "resample" in param and param["resample"] is not None:
        X = resample_axis(X, input_fs= param["input_fs"], output_fs= param["resample"])

    sensor_ids = [param["sensor_names"].index(x) for x in param["selected_sensors"]]
    X = X[:, :, sensor_ids, :]

    if "smooth" in param and param["smooth"] != None:
        for s in range(X.shape[2]):
            X[:, :, s, :] = np.apply_along_axis(func1d= moving_average, axis= 1, arr=X[:, :, s, :], w=param["smooth"])

    if "minmax_norm" in param and param["minmax_norm"]:
        X = normalize(X)
    if "znorm" in param and param["znorm"]:
        X = zscore(X, axis= 1)

    # ------------------------------------------ Generic--------------------------------------------------------------------------------
    # select classes
    if "classes" in param and param["classes"] is not None:
        # select certain classes from the data
        X, hcf, subjects, y = pick_classes(data = [X, hcf, subjects], y= y, classes = param["classes"], input_is_categorical= True)

    # ------------------------------------------ Augmentation---------------------------------------------------------------------------
    if (("aug_factor" in param) and (param["aug_factor"] is not None) and
        ("aug_method" in param) and (param["aug_method"] is not None)):
        aug_factor_type = type(param["aug_factor"])
        if (aug_factor_type != int) and (aug_factor_type!= float):
            raise ValueError(f"Param 'aug_factor' should be numeric but received '{param['aug_factor']}' with type '{aug_factor_type}'.")
        print(aug_factor_type)
        print("Initial Data shapes")
        print("X shape:", X.shape) #4D (1293, 2560, 1, 1)
        print("y shape:", y.shape) #2D (1293, 2)(After performing One hot encode)
        print("Shape of the Subjects: ",  subjects.shape)

        X_for_aug = X[:, :, :, 0] # 3D  (Converting 4D to 3D array-- (1293, 2560, 1)

        # convert from one-hot encoding
        y_for_aug = np.argmax(y, axis= 1) # 1D (1293)
        np.save('/home/abidhasan/Documents/DA_Project/preprocessed_data_for_GAN_Design/painmonit/X', X_for_aug)
        np.save('/home/abidhasan/Documents/DA_Project/preprocessed_data_for_GAN_Design/painmonit/y', y_for_aug)
        
        indices_0 = np.where(y_for_aug == 0)[0]
        indices_1 = np.where(y_for_aug == 1)[0]
        x_major = X_for_aug[indices_0]
        y_major = y_for_aug[indices_0]
        x_minor = X_for_aug[indices_1]
        y_minor = y_for_aug[indices_1]
        
        print("Shape of x major and minor data: ", x_major.shape, x_minor.shape) # 3D
        print("Shape of y major & minor indices: ", y_major.shape, y_minor.shape) #1D
        
        
        """
        # Find unique labels and their counts
        unique_labels, counts = np.unique(y_for_aug, return_counts=True)
        # Display the total quantity of each label
        for label, count in zip(unique_labels, counts):
            print(f"Label {label}: {count} occurrences")
        unique_labels, counts = np.unique(y_minor, return_counts=True)
        # Display the total quantity of each label
        for label, count in zip(unique_labels, counts):
            print(f"Label {label}: {count} occurrences")
        unique_labels, counts = np.unique(subjects, return_counts=True)
        # Display the total quantity of each label
        for label, count in zip(unique_labels, counts):
            print(f"Subject {label}: {count} occurrences")
        """


        if param["aug_method"] == "crop" or param["aug_method"] == "tw" or param["aug_method"] == "jitter" or  param["aug_method"] == "convolve" or param["aug_method"] == "rotation" or param["aug_method"] == "quantize" or param["aug_method"] == "drift":

            # To use "TSAUG" python DA library the infut format of x and y is 3D (Nr. of samples, Timesstamp, channels)
            # extend Dimension axis
            y_minor = np.expand_dims(y_minor, axis= -1)
            # repeat the value for the time series
            y_minor = np.repeat(y_minor, repeats= X.shape[1], axis=1)
            y_minor = np.expand_dims(y_minor, axis= -1)

            print("Minor Data (1) shapes before augmentation")
            print("X shape:", x_minor.shape) #3D ((252, 2560, 1))
            print("y shape:", y_minor.shape) #3D ((252, 2560, 1))

            if param["aug_factor"] <1:
                if param["aug_method"] == "crop" or param["aug_method"] == "slicing":
                    augmenter = (Crop(size= 1408) * 1 )
                elif param["aug_method"] == "tw":
                    augmenter = (TimeWarp() * 1 )
                elif param["aug_method"] == "jitter":
                    augmenter = (AddNoise(loc=0.0, scale=0.2, distr='gaussian', kind='additive') * 1)
                elif param["aug_method"] == "convolve" or param["aug_method"] == "magnitude_warping":
                    augmenter = (Convolve(window="flattop", size=16) * 1)
                elif param["aug_method"] == "rotation":
                    augmenter = (Reverse() @ 0.5 * 1)
                elif param["aug_method"] == "quantize":
                    augmenter = (Quantize(n_levels=[10, 20, 30]) * 1)
                elif param["aug_method"] == "drift" or param["aug_method"] == "scaling":
                    augmenter = (Drift(max_drift=(0.1, 0.5)) @ 0.8 * 1)

                x_aug, y_aug = augmenter.augment(x_minor, y_minor) # shape of X_aug,y_aug is (3D, 3D).

                mask = int(param["aug_factor"] * x_minor.shape[0])
                x_aug = x_aug[:mask] # 3D
                y_aug = y_aug[:mask] # 3D
                subjects_aug = subjects[:mask]

            else:
                if param["aug_method"] == "crop" or param["aug_method"] == "slicing":
                    augmenter = (Crop(size= 1408) * param["aug_factor"])
                elif param["aug_method"] == "jitter":
                    augmenter = (AddNoise(loc=0.0, scale=0.2, distr='gaussian', kind='additive') * param["aug_factor"])
                elif param["aug_method"] == "tw":
                    augmenter = (TimeWarp() * param["aug_factor"])
                elif param["aug_method"] == "convolve" or param["aug_method"] == "magnitude_warping":
                    augmenter = (Convolve(window="flattop", size=16) * param["aug_factor"])
                elif param["aug_method"] == "rotation":
                    augmenter = (Reverse() @ 0.5 * param["aug_factor"])
                elif param["aug_method"] == "quantize":
                    augmenter = (Quantize(n_levels=[10, 20, 30]) * param["aug_factor"])
                elif param["aug_method"] == "drift" or param["aug_method"] == "scaling":
                    augmenter = (Drift(max_drift=(0.1, 0.5)) @ 0.8 * param["aug_factor"])

                x_aug, y_aug = augmenter.augment(x_minor, y_minor) # shape of X_aug,y_aug is (3D, 3D).
                mask = int(param["aug_factor"] * x_minor.shape[0])
                subjects_aug = subjects[:mask]
                #subjects_aug = np.repeat(subjects, repeats= param["aug_factor"])

            x_aug = np.expand_dims(x_aug, axis= -1) #4D
            y_aug = to_categorical(y_aug[:, 0, 0]) #2D (After performing One hot encode)
            

        elif param["aug_method"] == "DGW" or param["aug_method"] == "RGW" or param["aug_method"] == "TW" or param["aug_method"] == "WW" or param["aug_method"] == "spawner" or param["aug_method"] == "permutation":

        # Calculate the number of samples to select (20% of the total samples)
            if param["aug_factor"] <1:
                mask = int(param["aug_factor"] * x_minor.shape[0])
                x_frac_aug = x_minor[:mask]
                y_frac_aug = y_minor[:mask]
                subjects_aug = subjects[:mask]
                #subjects = np.vstack([subjects, subjects_aug])
                if param["aug_method"] == "TW":
                    x_aug, y_aug = TW(x_frac_aug, y_frac_aug) # shape of x_frac_aug, y_frac_aug is (3D, 1D) and X_aug,y_aug is (3D, 1d).
                elif param["aug_method"] == "WW":
                    x_aug, y_aug = WW(x_frac_aug, y_frac_aug) # shape of x_frac_aug, y_frac_aug is (3D, 1D) and X_aug,y_aug is (3D, 1d).
                elif param["aug_method"] == "RGW":
                    x_aug, y_aug =  RGW(x_frac_aug, y_frac_aug) # shape of x_frac_aug, y_frac_aug is (3D, 1D) and X_aug,y_aug is (3D, 1d).
                elif param["aug_method"] == "DGW":
                    x_aug, y_aug =  DGW(x_frac_aug, y_frac_aug) # shape of x_frac_aug, y_frac_aug is (3D,1D) and X_aug,y_aug is (3D, 1d).
                elif param["aug_method"] == "spawner":
                    print(x_frac_aug.shape, y_frac_aug.shape)
                    x_aug, y_aug == spawner(x_frac_aug, y_frac_aug)
                    print(x_aug.shape, y_aug.shape)
                elif param["aug_method"] == "permutation":
                    x_aug, y_aug == permutation(x_frac_aug, y_frac_aug)
                    
                   
            elif param["aug_factor"] == 1:
                if param["aug_method"] == "TW":
                    x_aug, y_aug = TW(x_minor, y_minor) # shape of x_frac_aug, y_frac_aug(3D,1D) and X_aug,y_aug is (3D, 1d).
                elif param["aug_method"] == "WW":
                    x_aug, y_aug = WW(x_minor, y_minor) # shape of x_frac_aug, y_frac_aug(3D,1D) and X_aug,y_aug is (3D, 1d).
                elif param["aug_method"] == "RGW":
                    x_aug, y_aug =  RGW(x_minor, y_minor) # shape of x_frac_aug, y_frac_aug(3D,1D) and X_aug,y_aug is (3D, 1d).
                elif param["aug_method"] == "DGW":
                     x_aug, y_aug =  DGW(x_minor, y_minor) # shape of x_frac_aug, y_frac_aug(3D,1D) and X_aug,y_aug is (3D, 1d).
                elif param["aug_method"] == "spawner":
                    x_aug, y_aug = spawner(x_minor, y_minor)
                elif param["aug_method"] == "permutation":
                    x_aug, y_aug = permutation(x_minor, y_minor)
                mask = int(param["aug_factor"] * x_minor.shape[0])
                subjects_aug = subjects[:mask]
            
            elif param["aug_factor"] == 2:
                if param["aug_method"] == "TW":
                    x_aug_1, y_aug_1 = TW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = TW(x_minor, y_minor)
                elif param["aug_method"] == "WW":
                    x_aug_1, y_aug_1 = WW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = WW(x_minor, y_minor)
                elif param["aug_method"] == "RGW":
                    x_aug_1, y_aug_1 = RGW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = RGW(x_minor, y_minor)
                elif param["aug_method"] == "DGW":
                    x_aug_1, y_aug_1 = DGW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = DGW(x_minor, y_minor)
                elif param["aug_method"] == "spawner":
                    x_aug_1, y_aug_1 = spawner(x_minor, y_minor)
                    x_aug_2, y_aug_2 = spawner(x_minor, y_minor)
                elif param["aug_method"] == "permutation":
                    x_aug_1, y_aug_1 = permutation(x_minor, y_minor)
                    x_aug_2, y_aug_2 = permutation(x_minor, y_minor)
                
                x_aug = np.concatenate([x_aug_1, x_aug_2], axis=0)
                y_aug = np.concatenate([y_aug_1, y_aug_2], axis=0)
                mask = int(param["aug_factor"] * x_minor.shape[0])
                subjects_aug = subjects[:mask]
            
            elif param["aug_factor"] == 3:
                if param["aug_method"] == "TW":
                    x_aug_1, y_aug_1 = TW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = TW(x_minor, y_minor)
                    x_aug_3, y_aug_3 = TW(x_minor, y_minor)
                elif param["aug_method"] == "WW":
                    x_aug_1, y_aug_1 = WW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = WW(x_minor, y_minor)
                    x_aug_3, y_aug_3 = WW(x_minor, y_minor)
                elif param["aug_method"] == "RGW":
                    x_aug_1, y_aug_1 = RGW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = RGW(x_minor, y_minor)
                    x_aug_3, y_aug_3 = RGW(x_minor, y_minor)
                elif param["aug_method"] == "DGW":
                    x_aug_1, y_aug_1 = DGW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = DGW(x_minor, y_minor)
                    x_aug_3, y_aug_3 = DGW(x_minor, y_minor)
                elif param["aug_method"] == "spawner":
                    x_aug_1, y_aug_1 = spawner(x_minor, y_minor)
                    x_aug_2, y_aug_2 = spawner(x_minor, y_minor)
                    x_aug_3, y_aug_3 = spawner(x_minor, y_minor)
                elif param["aug_method"] == "permutation":
                    x_aug_1, y_aug_1 = permutation(x_minor, y_minor)
                    x_aug_2, y_aug_2 = permutation(x_minor, y_minor)
                    x_aug_3, y_aug_3 = permutation(x_minor, y_minor)
                    
                x_aug = np.concatenate([x_aug_1, x_aug_2, x_aug_3], axis=0)
                y_aug = np.concatenate([y_aug_1, y_aug_2, y_aug_3], axis=0)
                mask = int(param["aug_factor"] * x_minor.shape[0])
                subjects_aug = subjects[:mask]
                
            elif param["aug_factor"] == 4:
                if param["aug_method"] == "TW":
                    x_aug_1, y_aug_1 = TW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = TW(x_minor, y_minor)
                    x_aug_3, y_aug_3 = TW(x_minor, y_minor)
                    x_aug_4, y_aug_4 = TW(x_minor, y_minor)
                elif param["aug_method"] == "WW":
                    x_aug_1, y_aug_1 = WW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = WW(x_minor, y_minor)
                    x_aug_3, y_aug_3 = WW(x_minor, y_minor)
                    x_aug_4, y_aug_4 = WW(x_minor, y_minor)
                elif param["aug_method"] == "RGW":
                    x_aug_1, y_aug_1 = RGW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = RGW(x_minor, y_minor)
                    x_aug_3, y_aug_3 = RGW(x_minor, y_minor)
                    x_aug_4, y_aug_4 = RGW(x_minor, y_minor)
                elif param["aug_method"] == "DGW":
                    x_aug_1, y_aug_1 = DGW(x_minor, y_minor)
                    x_aug_2, y_aug_2 = DGW(x_minor, y_minor)
                    x_aug_3, y_aug_3 = DGW(x_minor, y_minor)
                    x_aug_4, y_aug_4 = DGW(x_minor, y_minor)
                elif param["aug_method"] == "spawner":
                    x_aug_1, y_aug_1 = spawner(x_minor, y_minor)
                    x_aug_2, y_aug_2 = spawner(x_minor, y_minor)
                    x_aug_3, y_aug_3 = spawner(x_minor, y_minor)
                    x_aug_4, y_aug_4 = spawner(x_minor, y_minor)
                elif param["aug_method"] == "permutation":
                    x_aug_1, y_aug_1 = permutation(x_minor, y_minor)
                    x_aug_2, y_aug_2 = permutation(x_minor, y_minor)
                    x_aug_3, y_aug_3 = permutation(x_minor, y_minor)
                    x_aug_4, y_aug_4 = permutation(x_minor, y_minor)
                x_aug = np.concatenate([x_aug_1, x_aug_2, x_aug_3, x_aug_4], axis=0)
                y_aug = np.concatenate([y_aug_1, y_aug_2, y_aug_3, y_aug_4], axis=0)
                mask = int(param["aug_factor"] * x_minor.shape[0])
                subjects_aug = subjects[:mask]
                
            print(x_aug.shape, y_aug.shape)         
            x_aug = np.expand_dims(x_aug, axis= -1) #4D
            y_aug = to_categorical(y_aug)           #2D (After performing One hot encode)
            
        elif param["aug_method"] == "GAN":
            mask = int(param["aug_factor"] * x_minor.shape[0])
            subjects_aug = subjects[:mask]
            x_aug = np.load(f'/home/abidhasan/Documents/DA_Project/BioVid/datasets/cGAN_Generated_data/painmonit/{aug_factor}_X.npy')
            y_aug = np.load(f'/home/abidhasan/Documents/DA_Project/BioVid/datasets/cGAN_Generated_data/painmonit/{aug_factor}_y.npy')
            print(x_aug.shape, y_aug.shape)         
            x_aug = np.expand_dims(x_aug, axis= -1) #4D
            y_aug = to_categorical(y_aug)           #2D (After performing One hot encode)
        


        print("Minor Data Shape after augmenation:")
        print("X shape:", x_aug.shape)
        print("y shape:", y_aug.shape)
        print("subjects shape:", subjects_aug.shape)

        #np.savetxt(f"/home/abidhasan/Documents/DA_Project/painmonit/datasets/augmented_data/x_{param['aug_method']}_{param['aug_factor']}_aug.txt", x_aug)
        #np.savetxt(f"/home/abidhasan/Documents/DA_Project/painmonit/datasets/augmented_data/y_{param['aug_method']}_aug.txt", y_aug)

        # TODO: probably you want to remove code related to "HCF" everywhere
        hcf = None

    return X, y, {"X": x_aug, "y": y_aug, "subjects": subjects_aug}, hcf, subjects

def conduct_experiment(X, y, subjects, clf, name, five_times= True):
    """ Method to conduct an experiment. Data to perform a ML task needs to be given.

    Args:
        X_cur (dataframe): X.
        y_cur (dataframe): y.
        subjects_cur (dataframe): subjects vector.
        clf (classifer): clf to use.
        name (str): Name of the file to save.
        five_times (bool, optional): Whether to conduct a 5x mean experiment. Defaults to False.
    """

    X, y, aug, hcf, subjects = prepare_data(X, y, subjects, clf.param)

    #print("X shape after preprocessing: ", X.shape)
    #print("y shape after preprocessing: ", y.shape)

    if hcf is not None:
        print("HCF shape after preprocessing: ", hcf.shape)

    if five_times:
        return five_loso(X, y, aug, hcf, subjects, clf, aug_method, aug_factor, output_csv = Path("results", "5_loso_{}.csv".format(name)))
    else:
        return loso_cross_validation(X, y, aug, hcf, subjects, clf, output_csv = Path("results", "{}.csv".format(name)))

def check_gpu():

    if 'linux' in platform.platform().lower():
        print("Check GPU...")
        if len(tf.config.list_physical_devices('GPU')) == 0:
            print("GPU is not available!")
            quit()

        print("GPU is available!")

if __name__ == "__main__":
    """Main function.
    """

    # set CWD to file location
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    #-------------------------------------------------------------------------------------------------------
    # Check if tensorflow is available
    #----------------------------------------------------------------------------------------------------
    check_gpu()

    #-------------------------------------------------------------------------------------------------------
    # Configuration begin
    #------------------------------------------------------------------------------------------------------
    # painmonit
   
    param= {
        "dataset": "painmonit",
        "resample": 256, # Give sampling_rate to resample to
        "selected_sensors": ["Eda_RB"],
        "classes": [[0], [4]],
    }

    sensor_names = []
    #-------------------------------------------------------------------------------------------------------
    # Configuration end
    #-------------------------------------------------------------------------------------------------------

    X, y, subjects = None, None, None

    if param["dataset"] == "biovid":
        X, y, subjects = read_biovid_np()
        param["sensor_names"] = biovid_sensors
        param["input_fs"] = 512
    elif param["dataset"] == "painmonit":
        param["painmonit_label"]= "covas" # or "heater"
        X, y, subjects = read_painmonit_np(label= param["painmonit_label"])
        param["sensor_names"] = painmonit_sensors
        param["input_fs"] = 250
    else:
        print("""Dataset '{}' is not available.
        Please choose either 'biovid' or 'painmonit' and make sure the according np files are created correctly.
        """.format(param["dataset"]))
        quit()

    assert len(X)==len(y)==len(subjects)

    print("\nDataset shape:")
    print("X.shape")
    print(X.shape)
    print("y.shape")
    print(y.shape)
    print("subjects.shape")
    print(subjects.shape)
    print("\n")
    
    # "crop", "jitter", "tw", "convolve", "rotation", "quantize", "drift", "TW", "WW" "RGW", "DGW", "spawner", "permutation", GAN

    # Deep learning
    param.update({"epochs": 100, "bs": 32, "lr": 0.0001, "smooth": 256, "resample": 256, "dense_out": 100, "minmax_norm": True})
    for clf in [mlp]:
        for aug_method in ["WW"]:
            for aug_factor in [.2, .4, .6, .8, 1, 2, 3, 4]:
                try:
                    param["aug_factor"] = aug_factor
                    param["aug_method"] = aug_method
                    conduct_experiment(X.copy(), y.copy(), subjects.copy(), clf= clf(param.copy()), name= param["dataset"], five_times= True)
                except Exception as e:
                    print(e)
                    