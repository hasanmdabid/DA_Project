# This projects will highlight the comparative analysis of GANS performance for both univariate and multivariate dataset
# ## 1st step will be to design an CNN model for the opportunity dataset and check the performance of the model without
# data augmentation.
#    ### 1. Preprocessing the dataset
#    ### 2. segmentation of the dataset
# ## 2nd step will be to use existing augmentation method to generate fabricated data and see the performance.
# ## 3rd step will be to formulate a unique evaluation matrix for the Augmentation methods (GANS)

import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numpy import mean
from numpy import std
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.utils import to_categorical


# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix


def load_data(x):
    x = pd.read_csv(
        f'/Users/mdabidhasan/PycharmProjects/Data_Augmentation/OPPORTUNITY/{x}.csv',
        dtype='float32')
    x = x.fillna(method="ffill")  # Replacing the 'Nan' by propagating last valid observation forward to next valid backfill
    #x = x[(~x.astype('bool')).mean(axis=1) < 0.10]  # Dropping any rows that contains 90% of its overall column values
    # is equal to 0

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

# data = pd.concat(
#    [s1r1, s1r2, s1r3, s1r4, s1r5, s1_drill, s2r1, s2r2, s2r3, s2r4, s2r5,
#     s2_drill], ignore_index=True)
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


################################################################################################
# The standard inout format for CNN is [SEGMENTS, TIME STAMP, FEATURES]
# Generating train and test data set
def Numpy_array(x):
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
        # Check the majority label ocurring in the considered window
        current_labels = labels[timestamp_idx:timestamp_idx + window_size]
        values, counts = np.unique(current_labels, return_counts=True)
        labels_to_save[segment_idx] = values[np.argmax(counts)]
        timestamp_idx += window_size
        segment_idx += 1
    return data_to_save, labels_to_save


# Creating the training and testingset of from f=created 3d Numpy array
# The formulation is adopted from "https://dl.acm.org/doi/10.5555/2832747.2832806"
trainxs1r1, trainys1r1 = Numpy_array(s1r1)
trainxs1r2, trainys1r2 = Numpy_array(s1r2)
trinxs1_drill, trainys1_drill = Numpy_array(s1_drill)
testxs1, testys1 = Numpy_array(s1r3)

trainxs2r1, trainys2r1 = Numpy_array(s2r1)
trainxs2r2, trainys2r2 = Numpy_array(s2r2)
trinxs2_drill, trainys2_drill = Numpy_array(s2_drill)
testxs2, testys2 = Numpy_array(s2r3)

trainxs3r1, trainys3r1 = Numpy_array(s3r1)
trainxs3r2, trainys3r2 = Numpy_array(s3r2)
trinxs3_drill, trainys3_drill = Numpy_array(s3_drill)
testxs3, testys3 = Numpy_array(s3r3)

trainxs4r1, trainys4r1 = Numpy_array(s4r1)
trainxs4r2, trainys4r2 = Numpy_array(s4r2)
trinxs4_drill, trainys4_drill = Numpy_array(s4_drill)
testxs4, testys4 = Numpy_array(s4r3)

trainx = np.concatenate((trainxs1r1, trainxs1r2, trinxs1_drill, trainxs2r1, trainxs2r2, trinxs2_drill, trainxs3r1,
                         trainxs3r2, trinxs3_drill, trainxs4r1, trainxs4r2, trinxs4_drill), axis=0)
trainy = np.concatenate((trainys1r1, trainys1r2, trainys1_drill, trainys2r1, trainys2r2, trainys2_drill, trainys3r1,
                         trainys3r2, trainys3_drill, trainys4r1, trainys4r2, trainys4_drill), axis=0)

testx = np.concatenate((testxs1, testxs2, testxs3, testxs4), axis=0)
testy = np.concatenate((testys1, testys2, testys3, testys4), axis=0)

print('shape of trainx =', trainx.shape)
print('shape of trainy =', trainy.shape)
print('Shape of testx =', testx.shape)
print('Shape of testy =', testy.shape)

################################################################################################
# The output data is defined as an integer for the class number. We must one hot encode these class
# integers so that the data is suitable for fitting a neural network multi-class classification model.
# We can do this by calling the to_categorical() Keras function.
# one hot encode y
trainy = to_categorical(trainy)
testy = to_categorical(testy)
print(trainx.shape, trainy.shape, testx.shape, testy.shape)


######################################################################################################

# Now the data is ready to be used in the 1D CNN model

#######################################################################################################

# Designing the 1D CNN model
# The model requires a three-dimensional input with [samples, time steps, features].
# Here one sample is one window of the time series data, each window has 60 time steps, and a time step has 107
# variables or features.The output for the model will be a s18-element vector containing the probability of a given
# window belonging to each of the 18 activity types.
# These input and output dimensions are required when fitting the model, and we can extract them from the provided
# training dataset.


# fit and evaluate a model
def evaluate_model(trainx, trainy, testx, testy, nr_filters, k_size):
    verbose, epochs, batch_size = 0, 100, 32
    n_timesteps, n_features, n_outputs = trainx.shape[1], trainx.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=nr_filters, kernel_size=k_size, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=nr_filters, kernel_size=k_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # fit network

    # help(EarlyStopping)
    # early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    model.fit(trainx, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testx, testy, batch_size=batch_size, verbose=0)
    # predictions = model.predict(testx).astype("int32")

    return accuracy


#######################################################################################################
# summarize scores
def summarize_results(scores, params):
    print(scores, params)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = mean(scores[i]), std(scores[i])
        print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
    # boxplot of scores
    pyplot.boxplot(scores, labels=params)
    pyplot.savefig('exp_cnn_filters.png')


################################################################################################
# Run an experiment function

def run_experiment(params, repeats):
    all_scores = list()
    for p in params:
        # repeat experiment
        scores = list()
        for r in range(repeats):
            score = evaluate_model(trainx, trainy, testx, testy, p, 3)
            score = score * 100.0
            print('>p=%d #%d: %.3f' % (p, r + 1, score))
            scores.append(score)
        all_scores.append(scores)
    # summarize results
    summarize_results(all_scores, params)


################################################################################################

################################################################################################
# run the experiment
run_experiment([8, 16, 32, 64, 128, 256], 3)


##################################################################################################

