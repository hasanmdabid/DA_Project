# This script will demonstrate the following programms:
# 1. Data Loading the data from oppurtunity data set.
# 2. Preparinf the data frame into 3D Numpy array[SAMPLE, TIMESTAMP, FEATURES]
# 3. Spliting the data 3D data set, into train and test set by Stratified K fold algorithm
# 4. 1D CNN model design
# 5. Model Evaluation
# 6.

###################################################################


# Importing all the required classes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


def load_data(x):
    x = pd.read_csv(
        f'/Users/mdabidhasan/Documents/Project In DAAD by Md Abid Hasan/Data Augmentation/Data Augmentation Project/Data/OPPORTUNITY/{x}.csv',
        dtype='float32')
    x = x.fillna(method='ffill')  # Replacing the 'Nan' values with 0 in the dataset
    # x = x[(~x.astype('bool')).mean(axis=1) < 0.10]  # Dropping any rows that contains 90% of its overall column values is equal to 0

    return x


s1r1 = load_data("S1-ADL1")
s1r2 = load_data("S1-ADL2")
s1r3 = load_data("S1-ADL3")
s1r4 = load_data("S1-ADL4")
s1r5 = load_data("S1-ADL5")
s1_drill = load_data("S1-Drill")
s2r1 = load_data("S2-ADL1")
s2r2 = load_data("S2-ADL2")
s2r3 = load_data("S2-ADL3")
s2r4 = load_data("S2-ADL4")
s2r5 = load_data("S2-ADL5")
s2_drill = load_data("S2-Drill")
s3r1 = load_data("S3-ADL1")
s3r2 = load_data("S3-ADL2")
s3r3 = load_data("S3-ADL3")
s3r4 = load_data("S3-ADL4")
s3r5 = load_data("S3-ADL5")
s3_drill = load_data("S1-Drill")
s4r1 = load_data("S4-ADL1")
s4r2 = load_data("S4-ADL2")
s4r3 = load_data("S4-ADL3")
s4r4 = load_data("S4-ADL4")
s4r5 = load_data("S4-ADL5")
s4_drill = load_data("S4-Drill")


def column_notation(data):
    data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                    '35',
                    '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51',
                    '52',
                    '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68',
                    '69',
                    '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85',
                    '86',
                    '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102',
                    '103', '104', '105', '106', '107', 'Activity Label']
    data['Activity Label'] = data['Activity Label'].replace(
        [406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511, 406508, 404508,
         408512, 407521, 405506], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    return data


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
s4r1 = column_notation(s4r1)
s4r2 = column_notation(s4r2)
s4r3 = column_notation(s4r3)
s4r4 = column_notation(s4r4)
s4r5 = column_notation(s4r5)
s4_drill = column_notation(s4_drill)

##########################################################################

# Activities list
activities = {
    'Old label': [0, 406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511,
                  406508, 404508, 408512, 407521, 405506],
    'New label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'Activity': ['No activity', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'close Door 2', 'Open fridge',
                 'Close fridge',
                 'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2',
                 'Close Drawer 2',
                 'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup', 'Toggle Switch']}
activities = pd.DataFrame(activities)
print(activities)
ACTIVITY = ['No activity', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'close Door 2', 'Open fridge',
            'Close fridge',
            'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2',
            'Close Drawer 2',
            'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup', 'Toggle Switch']

#######################################################################################################################
##   Exploratory Data Analysis
# checking any missing values through heatmap
sns.set_style('whitegrid')
sns.heatmap(s1r1.isnull(), cmap='viridis')
plt.savefig('Checking for missing data.png', dpi=200)

# Plotting the number of counts of each classes

count_classes = pd.value_counts(s1r1['Activity Label'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.savefig('Count numbers of each class.png', dpi=200)


########################################################################################################################
# Solution of imbalance problem
# I think this is the place where we should implement our augmentation techniques (Up-Sampling) for generating more
# minority class. Down SAMPLING techniques have some disadvantage i.e
#                1. High change that we will lose our important information from the dataset
#                2. Is good for univariate classification with big dataset.

########################################################################################################################
# Formatting the data into 3d Numpy array
def numpy_array(x):
    df = x
    data_and_labels = df.to_numpy()
    np_data = data_and_labels[:, :-1]  # All columns except the last one
    labels = data_and_labels[:, -1]  # The last column
    labels = labels.astype('int')  # Convert labels to int to avoid typing issues

    nb_timestamps, nb_sensors = np_data.shape
    window_size = 60  # Size of the data segments
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
        # Check the majority label occurring in the considered window
        current_labels = labels[timestamp_idx:timestamp_idx + window_size]
        values, counts = np.unique(current_labels, return_counts=True)
        labels_to_save[segment_idx] = values[np.argmax(counts)]
        timestamp_idx += window_size
        segment_idx += 1
    return data_to_save, labels_to_save


inputs, targets = numpy_array(s1r1)
print('data shape of 2D array', s1r1.shape)
print('data shape of inputs', inputs.shape)
print('data shape of targets before categorical', targets.shape)
##############################################################################################################
targets = to_categorical(targets)
print('data shape of targets after categorical', targets.shape)
##############################################################################################################
# Calling the K fold
# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

###############################################################################################################

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# K-fold Cross Validation model evaluation

fold_no = 1
verbose, epochs, batch_size = 0, 50, 32
for train, test in kfold.split(inputs, targets):
    # Define the model architecture

    n_timesteps, n_features, n_outputs = inputs.shape[1], inputs.shape[2], targets.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # fit network

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(inputs[train], targets[train],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
