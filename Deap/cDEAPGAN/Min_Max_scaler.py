import sys
import warnings

# Disabling the Python Warning.
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Data manipulation
import numpy as np  # for data manipulation

print('numpy: %s' % np.__version__)  # print version

# Visualization
import matplotlib
import matplotlib.pyplot as plt  # for data visualizationa

print('matplotlib: %s' % matplotlib.__version__)  # print version
import graphviz  # for showing model diagram

print('graphviz: %s' % graphviz.__version__)  # print version

import sys
import warnings

# Disabling the Python Warning.
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pandas as pd
import math

# Load Sample Opportunity data


s1r1 = pd.read_csv('S1-ADL1.csv')
s1r2 = pd.read_csv('S1-ADL2.csv')
s1r3 = pd.read_csv('S1-ADL3.csv')
s1r4 = pd.read_csv('S1-ADL4.csv')
s1r5 = pd.read_csv('S1-ADL5.csv')
s1_drill = pd.read_csv('S1-Drill.csv')
s2r1 = pd.read_csv('S2-ADL1.csv')
s2r2 = pd.read_csv('S2-ADL2.csv')
s2r3 = pd.read_csv('S2-ADL3.csv')
s2r4 = pd.read_csv('S2-ADL4.csv')
s2r5 = pd.read_csv('S2-ADL5.csv')
s2_drill = pd.read_csv('S1-Drill.csv')
s3r1 = pd.read_csv('S3-ADL1.csv')
s3r2 = pd.read_csv('S3-ADL2.csv')
s3r3 = pd.read_csv('S3-ADL3.csv')
s3r4 = pd.read_csv('S3-ADL4.csv')
s3r5 = pd.read_csv('S3-ADL5.csv')
s3_drill = pd.read_csv('S3-Drill.csv')


def column_notation(data):
    data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                    '18',
                    '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                    '35',
                    '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51',
                    '52',
                    '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68',
                    '69',
                    '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85',
                    '86',
                    '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101',
                    '102',
                    '103', '104', '105', '106', '107', 'Activity_Label']
    data['Activity_Label'] = data['Activity_Label'].replace([406516, 406517, 404516, 404517, 406520, 404520, 406505,
                                                             404505, 406519, 404519, 406511, 404511, 406508, 404508,
                                                             408512, 407521, 405506], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                                                       11, 12, 13, 14, 15, 16, 17])
    return data


def slided_numpy_array(data):
    import numpy as np
    x = data.to_numpy()

    # This function will generate the
    def get_strides(a, L, ov):
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])

        return np.array(out)

    L = 32  # Here L represent the Number of Samples of each DATA frame
    ov = 16  # ov represent the Sliding Window ration %%% Out of 32 the slided

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


# Opportunity have 18 Classes Including the No activity lavel(0).

s1r1 = column_notation(s1r1)
s1r2 = column_notation(s1r2)
s1r3 = column_notation(s1r3)
s1r4 = column_notation(s1r4)
s1r5 = column_notation(s1r5)
s1_drill = column_notation(s1_drill)
s2r1 = column_notation(s2r1)
s2r2 = column_notation(s2r2)
s2r3 = column_notation(s2r3)
s2r4 = column_notation(s2r4)
s2r5 = column_notation(s2r5)
s2_drill = column_notation(s2_drill)
s3r1 = column_notation(s3r1)
s3r2 = column_notation(s3r2)
s3r3 = column_notation(s3r3)
s3r4 = column_notation(s3r4)
s3r5 = column_notation(s3r5)
s3_drill = column_notation(s3_drill)


def Numpy_array(x):
    df = x
    data_and_labels = df.to_numpy()
    np_data = data_and_labels[:, :-1]  # All columns except the last one
    labels = data_and_labels[:, -1]  # The last column
    labels = labels.astype('int')  # Convert labels to int to avoid typing issues

    nb_timestamps, nb_sensors = np_data.shape
    window_size = 32  # Size of the data segments
    timestamp_idx = 0  # Index along the timestamp dimension
    segment_idx = 0  # Index for the segment dimension

    # Initialise the result arrays
    nb_segments = int(math.floor(nb_timestamps / window_size))
    print('Starting segmentation with a window size of %d resulting in %d segments and number of features is %d  ...' %
          (window_size, nb_segments, nb_sensors))
    data_to_save = np.zeros((nb_segments, window_size, nb_sensors), dtype=np.float32)
    labels_to_save = np.zeros(nb_segments, dtype=int)
    print('Dimension and shape of the generated blank numpy array')

    while segment_idx < nb_segments:
        data_to_save[segment_idx] = np_data[timestamp_idx:timestamp_idx + window_size, :]
        # Check the majority label ocurring in the considered window
        current_labels = labels[timestamp_idx:timestamp_idx + window_size]
        values, counts = np.unique(current_labels, return_counts=True)
        labels_to_save[segment_idx] = values[np.argmax(counts)]
        timestamp_idx += window_size
        segment_idx += 1
    return data_to_save, labels_to_save


# Checking the Number of Labels in the Dataset
# print(s1r1['Activity_Label'].value_counts())

trainxs1r1, trainys1r1 = slided_numpy_array(s1r1)
trainxs1r2, trainys1r2 = slided_numpy_array(s1r2)
trainxs1r3, trainys1r3 = slided_numpy_array(s1r3)
trainxs1_drill, trainys1_drill = slided_numpy_array(s1_drill)
trainxs1r4, trainys1r4 = slided_numpy_array(s1r4)
trainxs1r5, trainys1r5 = slided_numpy_array(s1r5)
trainxs2r1, trainys2r1 = slided_numpy_array(s2r1)
trainxs2r2, trainys2r2 = slided_numpy_array(s2r2)
trainxs2_drill, trainys2_drill = slided_numpy_array(s2_drill)
trainxs3r1, trainys3r1 = slided_numpy_array(s3r1)
trainxs3r2, trainys3r2 = slided_numpy_array(s3r2)
trainxs3_drill, trainys3_drill = slided_numpy_array(s3_drill)

trainx = np.concatenate(
    (trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
     trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)
trainy = np.concatenate(
    (trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
     trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)


# ----------------------------------------Without Majority Class-----------------------------------------------------------
def minority_class_only(x, y):
    global trainx, trainy
    idx = np.where(y == 0)
    for i in idx:
        trainx = np.delete(x, [i], axis=0)
        trainy = np.delete(y, [i], axis=0)

    return trainx, trainy


trainx_min, trainy_min = minority_class_only(trainx, trainy)

trainy_min = trainy_min.reshape(len(trainy), 1)  # **********************************************Reshaping it to (row*1)
nr_samples, nr_rows, nr_columns = trainx_min.shape
print('Shape of trainx:', trainx_min.shape)
print('Shape of trainy:', trainy_min.shape)

import collections
unique, counts = np.unique(trainy_min, return_counts=True)
counter = collections.Counter(trainy)
print(counter)
plt.bar(counter.keys(), counter.values())
plt.savefig('Bar_plot_of_Class_Distribution', dpi=400)

########################################################################################################################
# Preparing the dataset to use for the CGAN
# Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
# Here I am Using The MinMaxScaler to normalize the data
# 1st I have reshaped the 3D trainx to 2D then perform the MinMaxsacler
# then I reshape it to 4D (Nr_samples, Nr_rows, Nr_columns, Nr_channels)
# Here channels = 1 (For Image channels = 3 for Colored image and 1 for the Gray scale Image)
# The dimension the of the Trainx is 4d (Sample, Row, column, channels) and labels is Column Vector (Samples, 1)

from sklearn.preprocessing import MinMaxScaler

trainx_min = trainx_min.reshape(nr_samples * nr_rows, nr_columns)  # To sclae the data converting from 2D o 3D
scaler = MinMaxScaler(feature_range=(-1, 1))  # Declaring the scaler
scaler.fit(trainx_min)  # Fitting the scaler
trainx_min_scaled = scaler.transform(trainx_min)  # Transforming the DATA

from joblib import dump
dump(scaler, 'minmax_scaler.bin', compress=True)


# --------- Checking whether all the Feature value is scaled between +1 and -1 ------------------------------------------
max_val = np.max(trainx_min_scaled)
min_val = np.min(trainx_min_scaled)
print('Maximume_value after scaled:', max_val)
print('Minimume_value after scaled:', min_val)