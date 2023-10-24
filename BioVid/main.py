import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 - all logs shown
# 1 - filter out INFO logs
# 2 - additionally filter out WARNING logs
# 3 - additionally filter out ERROR logs

import platform
import numpy as np
import pandas as pd
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
from scripts.evaluation import loso_cross_validation, five_loso, accuracy, from_categorical, rfe_loso
from scripts.data_handling import read_biovid_np, pick_classes, normalize, resample_axis, read_painmonit_np, normalize_features

from config import painmonit_sensors, biovid_sensors, sampling_rate_biovid, sampling_rate_painmonit

#-------------------------------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------------------------------

def prepare_data(X, y, subjects, param):

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

    # ------------------------------------------ Generic
    # select classes
    if "classes" in param and param["classes"] is not None:
        # select certain classes from the data
        X, hcf, subjects, y = pick_classes(data = [X, hcf, subjects], y= y, classes = param["classes"], input_is_categorical= True)

    # ------------------------------------------ Augmentation
    if (("aug_factor" in param) and (param["aug_factor"] is not None) and
        ("aug_method" in param) and (param["aug_method"] is not None)):
        aug_factor_type = type(param["aug_factor"])
        if (aug_factor_type != int) and (aug_factor_type!= float):
            raise ValueError(f"Param 'aug_factor' should be numeric but received '{param['aug_factor']}' with type '{aug_factor_type}'.")
        
        print("Initial Data shapes")
        print("X shape:", X.shape) #4D (3480,1408,1,1)
        print("y shape:", y.shape) #2D (3480, 2)(After performing One hot encode)

        X_for_aug = X[:, :, :, 0]

        # convert from one-hot encoding
        y_for_aug = np.argmax(y, axis= 1) #1D(3480,)
        # extend axis
        y_for_aug = np.expand_dims(y_for_aug, axis= -1)
        # repeat the value for the time series
        y_for_aug = np.repeat(y_for_aug, repeats= X.shape[1], axis=1)
        y_for_aug = np.expand_dims(y_for_aug, axis= -1)

        print("Data shapes before augmentation")
        print("X shape:", X_for_aug.shape) #3D (3480,1408,1)
        print("y shape:", y_for_aug.shape) #3D (3480,1408,1)
        
        if param["aug_method"] == "crop" or param["aug_method"] == "jitter"or param["aug_method"] == "timewarp" or  param["aug_method"] == "convolve" or param["aug_method"] == "rotation" or param["aug_method"] == "quantize" or param["aug_method"] == "drift":
            # TODO: implement more augmentation methods here options
            if param["aug_method"] == "crop":
                augmenter = (Crop(size= 1408) * param["aug_factor"]) 
            elif param["aug_method"] == "jitter":
                augmenter = (AddNoise(loc=0.0, scale=0.2, distr='gaussian', kind='additive') * param["aug_factor"])
            elif param["aug_method"] == "timewarp":
                augmenter = (TimeWarp() * param["aug_factor"])
            elif param["aug_method"] == "convolve":
                augmenter = (Convolve(window="flattop", size=16) * param["aug_factor"])
            elif param["aug_method"] == "rotation":
                augmenter = (Reverse() @ 0.5 * param["aug_factor"])
            elif param["aug_method"] == "quantize":
                augmenter = (Quantize(n_levels=[10, 20, 30]) * param["aug_factor"])
            elif param["aug_method"] == "drift":
                augmenter = (Drift(max_drift=(0.1, 0.5)) @ 0.8 * param["aug_factor"])
            
            x_aug, y_aug = augmenter.augment(X_for_aug, y_for_aug)
            x_aug = np.expand_dims(x_aug, axis= -1)
            y_aug = to_categorical(y_aug[:, 0, 0])
        
        elif param["aug_method"] == "DGW" or param["aug_method"] == "RGW" or param["aug_method"] == "TW":
            #-------------*********preparing the data for the DTW algorithm (Input is X_aug(3D), y(1D), output is X(3D), Y(1D))************** --------------------------
            x_2d = X_for_aug.reshape(-1, X_for_aug.shape[-1]) #2D  (4899840, 1)
            #print("x after converting 3d to 2d shape:", x_2d.shape)
            y_2d = y_for_aug.reshape(-1, 1) #2D  (4899840, 1)
            #print("y after converting 3d to 2d shape:", y_2d.shape)
            dataset = np.concatenate((x_2d, y_2d), axis=1)
            #print('Shape of the dataset=', dataset.shape) # 2D (4899840, 2)
            
            #-----------------------------------------Selectinf a certain amount of fractional data------------------------
            Aug_frac = param["aug_factor"]  # Select the values of the augmatation fraction.
            print('Augmentation Factor', Aug_frac)   
            n_ecp_samples = math.ceil(dataset.shape[0]*Aug_frac) 
            print('Shape of Number of Expected samples:', n_ecp_samples)

            #---------------------------------------------------------------- Performing the Downsample With SKlearn----------------------------------------------------------------
            # We applied resample() method from the sklearn.utils module for downsampling, The replace = True attribute performs random resampling with replacement. The n_samples attribute 
            # defines the number of records you want to select from the original records. We have set the value of this attribute to the number of records in the spam dataset so the two sets will be balanced.
            Aug_downsample = resample(dataset, replace=True, n_samples=n_ecp_samples)
            print('Shape of Randomly selected data', Aug_downsample.shape)
            
            def slided_numpy_array(data, L, ov = 0 ):
                import numpy as np
                # This function will generate the 3D Data for the DEEP CNN model
                # Input is a 2D array where the last column contains the labels information
                # x = data.to_numpy()
                def get_strides(a, L, ov):
                    out = []

                    for i in range(0, a.shape[0] - L + 1, L - ov):
                        out.append(a[i : i + L, :])
                    return np.array(out)
                x = get_strides(data, L, ov)
                segment_idx = 0  # Index for the segment dimension
                nb_segments, nb_timestamps, nb_columns = x.shape
                data_to_save = np.zeros((nb_segments, nb_timestamps, nb_columns - 1), dtype=np.float32)
                labels_to_save = np.zeros(nb_segments, dtype=int)
                
                for i in range(0, nb_segments):
                    data_3D = x[i][:][:]
                    data_to_save[i] = data_3D[:, :-1]
                    labels = data_3D[:, -1]
                    labels = labels.astype("int")  # Convert labels to int to avoid typing issues
                    values, counts = np.unique(labels, return_counts=True)
                    labels_to_save[i] = values[np.argmax(counts)]

                return data_to_save, labels_to_save
            
            x_frac_aug, y_frac_aug = slided_numpy_array(Aug_downsample, L= X_for_aug.shape[1], ov=0 )
            #print('Shape of x for DTW input', x_frac_aug.shape)
            #print('Shape of y for DTW input', y_frac_aug.shape)
            # Augmenting the fractional data
            if param["aug_method"] == "DGW":
                x_aug, y_aug =  DGW(x_frac_aug, y_frac_aug) # shape of X_aug,y_aug is (3D, 1d). 
            elif param["aug_method"] == "RGW":
                x_aug, y_aug =  RGW(x_frac_aug, y_frac_aug) # shape of X_aug,y_aug is (3D, 1d).
            elif param['TW'] == TW:
                x_aug, y_aug = TW(x_frac_aug, y_frac_aug) # shape of X_aug,y_aug is (3D, 1d).
            
            x_aug = np.expand_dims(x_aug, axis= -1)
            y_aug = to_categorical(y_aug)

        else:
            raise NotImplementedError(f"Augmentation method '{param['aug_method']}' is not available.")
       
       #-------------------------------------Conver the List object into Numpy arrays --------------------------------

        print("Data Shape after augmenation:")
        print("X shape:", x_aug.shape)
        print("y shape:", y_aug.shape)
        
        #np.savetxt(f"/home/abidhasan/Documents/DA_Project/BioVid/datasets/augmented_data/x_{param['aug_method']}_{param['aug_factor']}_aug.txt", x_aug)
        #np.savetxt(f"/home/abidhasan/Documents/DA_Project/BioVid/datasets/augmented_data/y_{param['aug_method']}_aug.txt", y_aug)

        # extend subjects accordingly
        # TODO: check if this is correct
        print('Data type of the Subjects:', type(subjects))
        unique_valence, counts_valence = np.unique(subjects, return_counts=True)
        print('Count of Subject:', np.asarray((unique_valence, counts_valence)).T)
        subjects_aug = np.repeat(subjects, repeats= param["aug_factor"])
        unique_valence, counts_valence = np.unique(subjects_aug, return_counts=True)
        print('Count of Subject:', np.asarray((unique_valence, counts_valence)).T)
        print("shape of subject: ", subjects.shape)
        print("shape of augmented subjects: ", subjects_aug.shape)

        # concatenate
        X = np.concatenate([X, x_aug], axis= 0)
        y = np.concatenate([y, y_aug], axis= 0)
        subjects = np.concatenate([subjects, subjects_aug], axis= 0)

        print("Data Shape after concatination with original and augmented data:")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("subjects shape:", subjects.shape)

        hcf = None

    return X, y, hcf, subjects

def conduct_experiment(X, y, subjects, clf, name, five_times= True, rfe= False):
    """ Method to conduct an experiment. Data to perform a ML task needs to be given.

    Args:
        X_cur (dataframe): X.
        y_cur (dataframe): y.
        subjects_cur (dataframe): subjects vector.
        clf (classifer): clf to use.
        name (str): Name of the file to save.
        five_times (bool, optional): Whether to conduct a 5x mean experiment. Defaults to False.
    """

    X, y, hcf, subjects = prepare_data(X, y, subjects, clf.param)

    print("X shape after preprocessing: ", X.shape)
    print("y shape after preprocessing: ", y.shape)

    if hcf is not None:
        print("HCF shape after preprocessing: ", hcf.shape)

    if rfe:
        return rfe_loso(X, y, hcf, subjects, clf)
    else:
        if five_times:
            return five_loso(X, y, hcf, subjects, clf, output_csv = Path("results", "5_loso_{}.csv".format(name)))
        else:
            return loso_cross_validation(X, y, hcf, subjects, clf, output_csv = Path("results", "{}.csv".format(name)))

def check_simple_metrics():
    print("Checking simplistic metrics...")

    df = pd.DataFrame([])

    # read in biovid
    param = {
        "dataset": "biovid",
        "input_fs": sampling_rate_biovid,
        "sensor_names": biovid_sensors,
        "selected_sensors": ["gsr"],
        "classes": [[0], [4]],
    }
    x_biovid, y_biovid, subjects_biovid = read_biovid_np()
    x_biovid, _, hcf, y_biovid, subjects_biovid = prepare_data(x_biovid, y_biovid, subjects_biovid, param)
    x_biovid = x_biovid.copy() # make variable available inside functions that are defined here
    y_biovid = from_categorical(y_biovid)

    # read in painmonit
    param["dataset"]= "painmonit"
    param["input_fs"] = sampling_rate_painmonit
    param["sensor_names"] = painmonit_sensors
    param["painmonit_label"]= "heater"
    param["classes"]= [[0], [5]]
    param["selected_sensors"]= ["Eda_RB"]
    x_painmonit = None
    x_painmonit, y_painmonit, subjects_painmonit = read_painmonit_np(label= param["painmonit_label"])
    x_painmonit, _, hcf, y_painmonit, subjects_painmonit = prepare_data(x_painmonit, y_painmonit, subjects_painmonit, param)
    x_painmonit = x_painmonit.copy() # make variable available inside functions that are defined here
    y_painmonit = from_categorical(y_painmonit)

    def evaluate_metric(metric_str):
        """Function to evaluate a metric on the painmonit and biovid dataset.
        The given `metric_str` will be applied in the template 'f"[{metric_str} for x in dataset[:, :, 0, 0]]"'.
        Results will be saved in the df at locations '["Painmonit", metric_str]' and '["Biovid", metric_str]'.

        Args:
            metric_str (string): The metric to check. For example, "x[0] < x[-1]".
        """
        x_painmonit # make variable available for 'eval' call
        x_biovid # make variable available for 'eval' call

        # evaluate painmonit
        pred_painmonit = eval(f"[{metric_str} for x in x_painmonit[:, :, 0, 0]]")
        acc_painmonit = round(accuracy(pred_painmonit, y_painmonit) * 100, 2)

        # evaluate biovid
        pred_biovid = eval(f"[{metric_str} for x in x_biovid[:, :, 0, 0]]")
        acc_biovid = round(accuracy(pred_biovid, y_biovid) * 100, 2)
    
        # save results
        df.loc["Painmonit", metric_str] = acc_painmonit
        df.loc["Biovid", metric_str] = acc_biovid

    evaluate_metric(metric_str= "x[0] < x[-1]")
    evaluate_metric(metric_str= "len(x) * (7/10) < np.argmax(x)")
    evaluate_metric(metric_str= "len(x) * (1/3) < np.argmax(x) - np.argmin(x)")
    evaluate_metric(metric_str= "len(x) * (1/4) < np.argmax(x) - np.argmin(x)")
    evaluate_metric(metric_str= "0 < sum(x[1:] - x[:-1])")

    # save table
    print(df)
    df.to_csv(Path("results", "simple_metrics.csv"), sep= ";", decimal= ",")

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

    # Simple metrics
    #check_simple_metrics()

    #-------------------------------------------------------------------------------------------------------
    # Check if tensorflow is available
    #----------------------------------------------------------------------------------------------------
    check_gpu()

    #-------------------------------------------------------------------------------------------------------
    # Configuration begin
    #------------------------------------------------------------------------------------------------------
    # biovid
    param= {
        "dataset": "biovid",
        "resample": 256, # Give sampling_rate to resample to
        "selected_sensors": ["gsr"],
        "classes": [[0], [4]],
        #"aug": ["discriminative_guided_warp"]
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
        param["painmonit_label"]= "heater" #or "covas"
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

       # Deep learning
    param.update({"epochs": 100, "bs": 32, "lr": 0.0001, "smooth": 256, "resample": 256, "dense_out": 100, "minmax_norm": True})

    for clf in [mlp]:
         for aug_method in ["RGW"]:
            for aug_factor in [0.2, 1, 0.2]:
                try:
                    param["aug_factor"] = aug_factor
                    param["aug_method"] = aug_method
                    conduct_experiment(X.copy(), y.copy(), subjects.copy(), clf= clf(param.copy()), name= param["dataset"], five_times= True, rfe= False)
                except Exception as e:
                    print(e)