
import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import windowed_numpy_3Darray
import nn_models
import gc
import utils.augmentation as aug
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
import nn_models
from nn_models import *


s1r1 = load_data.load_data("S1-ADL1")
s1r2 = load_data.load_data("S1-ADL2")
s1r3 = load_data.load_data("S1-ADL3")
s1r4 = load_data.load_data("S1-ADL4")
s1r5 = load_data.load_data("S1-ADL5")
s1_drill = load_data.load_data("S1-Drill")
s2r1 = load_data.load_data("S2-ADL1")
s2r2 = load_data.load_data("S2-ADL2")
s2r3 = load_data.load_data("S2-ADL3")
s2r4 = load_data.load_data("S2-ADL4")
s2r5 = load_data.load_data("S2-ADL5")
s2_drill = load_data.load_data("S2-Drill")
s3r1 = load_data.load_data("S3-ADL1")
s3r2 = load_data.load_data("S3-ADL2")
s3r3 = load_data.load_data("S3-ADL3")
s3r4 = load_data.load_data("S3-ADL4")
s3r5 = load_data.load_data("S3-ADL5")
s3_drill = load_data.load_data("S1-Drill")
s4r1 = load_data.load_data("S4-ADL1")
s4r2 = load_data.load_data("S4-ADL2")
s4r3 = load_data.load_data("S4-ADL3")
s4r4 = load_data.load_data("S4-ADL4")
s4r5 = load_data.load_data("S4-ADL5")
s4_drill = load_data.load_data("S4-Drill")
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
        data['Activity_Label'] = data['Activity_Label'].replace(
            [406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511, 406508,
             404508,
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


#-------------------------------------------------->Activities list<-----------------------------------------------------------
activities = {'Old label': [0, 406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511,
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
    guided_TW_1 = aug.time_warp(data_min_3D)
    guided_TW_2 = aug.time_warp(data_min_3D)
    guided_TW_3 = aug.time_warp(data_min_3D)
    guided_TW_4 = aug.time_warp(data_min_3D)
    guided_TW_5 = aug.time_warp(data_min_3D)
    guided_TW_6 = aug.time_warp(data_min_3D)
    guided_TW_7 = aug.time_warp(data_min_3D)
    guided_TW_8 = aug.time_warp(data_min_3D)
    guided_TW_9 = aug.time_warp(data_min_3D)
    guided_TW_10 = aug.time_warp(data_min_3D)
    
    data_hybrid = np.concatenate((data_Maj_3D, data_min_3D, guided_TW_1, guided_TW_2, guided_TW_3, guided_TW_4, guided_TW_5, guided_TW_6, guided_TW_7, guided_TW_8, guided_TW_9, guided_TW_10), axis=0) 
    #labels_hybrid = np.concatenate((labels_maj_to_save.reshape(labels_maj_to_save.shape[0], 1), labels_min_to_save.reshape(labels_min_to_save.shape[0], 1),labels_min_to_save.reshape(labels_min_to_save.shape[0], 1)), axis=0)
    labels_hybrid = np.concatenate((labels_maj_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, 
                                    labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save), axis=0)                              
       
    return data_hybrid, labels_hybrid

def RGW(data):
    ##========================Seperatiing the Null Class from the data to generate Only the Minority class============
    data_maj = data.loc[data.Activity_Label == 0]
    #print('Shape of DATA with Majority target', data_maj.shape)
    
    data_maj = data_maj.to_numpy()

    data_min = data.loc[data.Activity_Label > 0]
    #print('Shape of DATA with Minority target', data_min.shape)
    data_min = data_min.to_numpy()
    
    data_Maj_3D, labels_maj_to_save = slided_numpy_array(data_maj)
    data_min_3D, labels_min_to_save = slided_numpy_array(data_min) 
    guided_RGW_1 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    guided_RGW_2 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    guided_RGW_3 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    guided_RGW_4 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    guided_RGW_5 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    guided_RGW_6 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    guided_RGW_7 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    guided_RGW_8 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    guided_RGW_9 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    guided_RGW_10 = aug.random_guided_war(data_min_3D, labels_min_to_save)
    
    data_hybrid = np.concatenate((data_Maj_3D, data_min_3D, guided_RGW_1, guided_RGW_2, guided_RGW_3, guided_RGW_4, guided_RGW_5, guided_RGW_6, guided_RGW_7, guided_RGW_8, guided_RGW_9, guided_RGW_10), axis=0) 
    #labels_hybrid = np.concatenate((labels_maj_to_save.reshape(labels_maj_to_save.shape[0], 1), labels_min_to_save.reshape(labels_min_to_save.shape[0], 1),labels_min_to_save.reshape(labels_min_to_save.shape[0], 1)), axis=0)
    labels_hybrid = np.concatenate((labels_maj_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, 
                                    labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save), axis=0)                              
       
    return data_hybrid, labels_hybrid

def DGW(data):
    ##========================Seperatiing the Null Class from the data to generate Only the Minority class============
    data_maj = data.loc[data.Activity_Label == 0]
    #print('Shape of DATA with Majority target', data_maj.shape)
    
    data_maj = data_maj.to_numpy()

    data_min = data.loc[data.Activity_Label > 0]
    #print('Shape of DATA with Minority target', data_min.shape)
    data_min = data_min.to_numpy()
    
    data_Maj_3D, labels_maj_to_save = slided_numpy_array(data_maj)
    data_min_3D, labels_min_to_save = slided_numpy_array(data_min) 
    guided_DGW_1 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DGW_2 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DGW_3 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DGW_4 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DGW_5 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DGW_6 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DGW_7 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DGW_8 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DGW_9 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    guided_DGW_10 = aug.discriminative_guided_warp(data_min_3D, labels_min_to_save)
    
    data_hybrid = np.concatenate((data_Maj_3D, data_min_3D, guided_DGW_1, guided_DGW_2, guided_DGW_3, guided_DGW_4, guided_DGW_5, guided_DGW_6, guided_DGW_7, guided_DGW_8, guided_DGW_9, guided_DGW_10), axis=0) 
    #labels_hybrid = np.concatenate((labels_maj_to_save.reshape(labels_maj_to_save.shape[0], 1), labels_min_to_save.reshape(labels_min_to_save.shape[0], 1),labels_min_to_save.reshape(labels_min_to_save.shape[0], 1)), axis=0)
    labels_hybrid = np.concatenate((labels_maj_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, 
                                    labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save, labels_min_to_save), axis=0)                              
       
    return data_hybrid, labels_hybrid

# ----------------------------------------->Creating the data set<-----------------------------------------------------------------------------
trainxs1r1, trainys1r1 = DGW(s1r1)
trainxs1r2, trainys1r2 = DGW(s1r2)
trainxs1r3, trainys1r3 = DGW(s1r3)
trainxs1_drill, trainys1_drill = DGW(s1_drill)
trainxs1r4, trainys1r4 = DGW(s1r4)
trainxs1r5, trainys1r5 = DGW(s1r5)
trainxs2r1, trainys2r1 = DGW(s2r1)
trainxs2r2, trainys2r2 = DGW(s2r2)
trainxs2_drill, trainys2_drill = DGW(s2_drill)
trainxs3r1, trainys3r1 = DGW(s3r1)
trainxs3r2, trainys3r2 = DGW(s3r2)
trainxs3_drill, trainys3_drill = DGW(s3_drill)
valxs2r3, valys2r3 = windowed_numpy_3Darray.slided_numpy_array(s2r3)
valxs3r3, valys3r3 = windowed_numpy_3Darray.slided_numpy_array(s3r3)
testxs2r4, testys2r4 = windowed_numpy_3Darray.slided_numpy_array(s2r4)
testxs2r5, testys2r5 = windowed_numpy_3Darray.slided_numpy_array(s2r5)
testxs3r4, testys3r4 = windowed_numpy_3Darray.slided_numpy_array(s3r4)
testxs3r5, testys3r5 = windowed_numpy_3Darray.slided_numpy_array(s3r5)

# Generate the  synthetic data from the Generator
trainx = np.concatenate((trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
                 trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)

trainy = np.concatenate((trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
                 trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)

valx = np.concatenate((valxs2r3, valxs3r3), axis=0)

valy = np.concatenate((valys2r3, valys3r3), axis=0)

testx = np.concatenate((testxs2r4, testxs2r5, testxs3r4, testxs3r5), axis=0)
testy = np.concatenate((testys2r4, testys2r5, testys3r4, testys3r5), axis=0)
print('shape of trainx =', trainx.shape)
print('shape of trainy =', trainy.shape)
print('Shape of testx =', testx.shape)
print('Shape of testy =', testy.shape)
print('Shape of Valx=', valx.shape)
print('Shape of valy=', valy.shape)


print('Unique Indices in train set:', np.unique(trainy))
print('Unique indices in test  set:', np.unique(testy))
unique, counts = np.unique(trainy, return_counts=True)
print('Number of instances in train Y:')
print(np.asarray((unique, counts)).T)

##################################################################################################################
# The output data is defined as an integer for the class number. We must one hot encode these class
# integers so that the data is suitable for fitting a neural network multi-class classification model.
# We can do this by calling the to_categorical() Keras function.
# one hot encode y

trainy = to_categorical(trainy)
testy = to_categorical(testy)
valy = to_categorical(valy)
print(trainx.shape, trainy.shape, testx.shape, testy.shape, valx.shape, valy.shape)

# ----------------------- Assign the parameters of the model-------------------------------------------------
n_timesteps, n_features, n_outputs = trainx.shape[1], trainx.shape[2], trainy.shape[1]
filterSizes = [64, 48, 32, 16]
nkerns = [(11, 1), (10, 1), (6, 1), (5, 1)]
poolSizes = [(2, 1), (3, 1), (1, 1), (1, 1)]
inputMLP = [1000, 512]
activationConv = 'relu'
activationMLP = 'relu'

verbose = 1
epochs = 100
batch_size = 100

n_steps, n_length = 4, 8  # ******************* Because if the window size if 32 = 4*8   ***********************


# ---------------------Reshape the input according to the model------------------------------------------------

# Fit and Evaluate the Model

def createandevaluate(trainx, testx, valx, trainy, testy, valy,  n_timesteps, n_features, n_outputs, n_steps, n_length, nkerns, filterSizes,
                      poolSizes, activationConv, inputMLP, activationMLP, verbose, epochs, batch_size,
                      start_time=datetime.now()):
    global model, tranx_f, testx_f, valx_f, trainx_f
    method = 'random_guided_warp_50%_SLIDED WINDOW_9K'

    #modelname = input('Mention the Model:')        # Enter the name of the model that you want to test
    modelname = 'DEEPCONVLSTM'

    if modelname == 'NORMCONV1D':
        trainx_f = trainx
        testx_f = testx
        valx_f = valx
        model = nn_models.NORMCONV1D(n_timesteps, n_features, n_outputs, nkerns, filterSizes, poolSizes, activationConv,
                                     inputMLP, activationMLP, withBatchNormalization=True)

    elif modelname == 'CONV2D':
        trainx_f = trainx
        testx_f = testx
        valx_f = valx
        model = nn_models.CONV2D(n_timesteps, n_features, n_outputs, nkerns, filterSizes, activationConv, inputMLP,
                                 activationMLP, withBatchNormalization=True)


    elif modelname == 'DEEPCONVLSTM':
        trainx_f = trainx
        testx_f = testx
        valx_f = valx
        model = nn_models.DEEPCONVLSTM(n_timesteps, n_features, n_outputs, nkerns, filterSizes, poolSizes,
                                       activationConv, inputMLP, activationMLP)


    elif modelname == 'CONVLSTM_1D_HYBRID':
        trainx_f = trainx.reshape((trainx.shape[0], n_steps, n_length, n_features))
        testx_f = testx.reshape((testx.shape[0], n_steps, n_length, n_features))
        valx_f = valx.reshape((valx.shape[0], n_steps, n_length, n_features))
        model = nn_models.CONVLSTM_1D_NONHYBRID(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, inputMLP)

    elif modelname == 'CONVLSTM_1D':
        trainx_f = trainx.reshape((trainx.shape[0], n_steps, n_length, n_features))
        testx_f = testx.reshape((testx.shape[0], n_steps, n_length, n_features))
        valx_f = valx.reshape((valx.shape[0], n_steps, n_length, n_features))
        model = nn_models.CONVLSTM_1D(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, inputMLP,
                                      activationMLP, activationConv)

    elif modelname == 'CONVLSTM_2D_HYBRID':
        trainx_f = trainx.reshape((trainx.shape[0], n_steps, 1, n_length, n_features))
        testx_f = testx.reshape((testx.shape[0], n_steps, 1, n_length, n_features))
        valx_f = valx.reshape((valx.shape[0], n_steps, 1, n_length, n_features))
        model = nn_models.CONVLSTM_2D_NONHYBRID(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns,
                                                activationConv, activationMLP, inputMLP)

    elif modelname == 'CONVLSTM_2D':
        trainx_f = trainx.reshape((trainx.shape[0], n_steps, 1, n_length, n_features))
        testx_f = testx.reshape((testx.shape[0], n_steps, 1, n_length, n_features))
        valx_f = valx.reshape((valx.shape[0], n_steps, 1, n_length, n_features))
        model = nn_models.CONVLSTM_2D(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, activationConv,
                                      activationMLP, inputMLP)

    # Compiling The model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Model created")
    print(model.summary())

    # Fitting the model With the system.
    print("Fit model:")

    # Fit the model
    model.fit(trainx_f, trainy, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(valx_f, valy))
    # model.fit(trainx_f, trainy, batch_size=batch_size, epochs=epochs, verbose=verbose)

    # Evaluating the model
    print("Evaluate model: ")

    # Evaluate model
    loss, accuracy = model.evaluate(testx_f, testy, verbose=verbose)
    testing_pred = model.predict(testx_f)
    testing_pred = testing_pred.argmax(axis=-1)
    true_labels = testy.argmax(axis=-1)
    f1scores_per_class = (f1_score(true_labels, testing_pred, average=None))
    average_fscore_macro = (f1_score(true_labels, testing_pred, average="macro"))
    average_fscore_weighted = (f1_score(true_labels, testing_pred, average="weighted"))
    print('Average F1 Score per class:', f1scores_per_class)
    print('Average_Macro F1 Score of the Model:', average_fscore_macro)
    print('Average_Weighted F1 score of the Model:', average_fscore_weighted)
    print('Accuracy of the Model:', accuracy)

    # time_elapsed = datetime.now() - start_time

    saveResultsCSV(method, modelname, epochs, batch_size, accuracy, average_fscore_macro, average_fscore_weighted, "", "")

    tf.keras.backend.clear_session()

    del trainx_f, testx_f, valx_f, model, valx, valy, trainx, testx, trainy, testy

    gc.collect()


# -------------------------------------------------------------------------------------------------------------
createandevaluate(trainx, testx, valx, trainy, testy, valy,  n_timesteps, n_features, n_outputs, n_steps, n_length, nkerns, filterSizes,
                      poolSizes, activationConv, inputMLP, activationMLP, verbose, epochs, batch_size,
                      start_time=datetime.now())

