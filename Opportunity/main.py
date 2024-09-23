# To use theis repository for Opportunity dataset one must have to download the "opportunity" dataset from the following link.
# https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition
# The dataset is public and anyone can use it without any prior consent.
# To read the dataset, please save the csv files under teh following folders "./opportunity/Data/{x}.csv"
# This projects will highlight the comparative analysis of Augmentation Algorithms performance for both univariate and multivariate dataset
# ## 1st step will be to design an CNN model for the opportunity dataset and check the performance of the model without
# data augmentation.
#    ### 1. Preprocessing the dataset
#    ### 2. Performance Evaluation of the Augmentation Algorithms on the selected dataset.

# pylint: disable-all
import gc
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import f1_score
import train_test_split
import nn_models
from nn_models import *


# --------- Spliting the Data into Train and test split.
#  ['jitter', 'scaling', 'rotation', 'permutation', 'magnitude_warp', 'slicing', 'time_warp', 'window_warp', 'spawner','random_guided_warp', 'discriminative_guided_warp', 'cGAN']
factors = [0.2, 0.4, 0.6, 0.8, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
families = ['jitter', 'scaling', 'rotation', 'permutation', 'magnitude_warp', 'slicing','time_warp', 'window_warp', 'spawner', 'random_guided_warp', 'discriminative_guided_warp', 'cGAN']
#looping over Factors 
for factor in factors:
    #looping over Familiyes
    for family in families: 
        print(f"Evaluating the Model for the Factor = {factor} and the Family = {family}")
        family, trainx, trainy, valx, valy, testx, testy = train_test_split.train_test_split(factor, family)
        method = family
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
        print(trainx.shape, trainy.shape, testx.shape,
            testy.shape, valx.shape, valy.shape)

        # ----------------------- Assign the parameters of the model-------------------------------------------------
        n_timesteps, n_features, n_outputs = trainx.shape[1], trainx.shape[2], trainy.shape[1]
        filterSizes = [64, 48, 32, 16]
        nkerns = [(11, 1), (10, 1), (6, 1), (5, 1)]
        poolSizes = [(2, 1), (3, 1), (1, 1), (1, 1)]
        inputMLP = [1000, 512]
        activationConv = 'relu'
        activationMLP = 'relu'
        verbose = 2
        epochs = 300
        batch_size = 100
        # ******************* Because if the window size if 32 = 4*8   ***********************
        n_steps, n_length = 4, 8
        
        # Specify How many times do you want to rerun the test
        num_repeats = 5

        # ---------------------Reshape the input according to the model------------------------------------------------

        # Fit and Evaluate the Model


        def createandevaluate(method, factor, trainx, testx, valx, trainy, testy, valy,  n_timesteps, n_features, n_outputs, n_steps, n_length, nkerns, filterSizes,
                            poolSizes, activationConv, inputMLP, activationMLP, verbose, epochs, batch_size, start_time=datetime.now()):
            
            # modelname = input('Mention the Model:')        # Enter the name of the model that you want to test
            modelname = 'DEEPCONVLSTM'

            if modelname == 'NORMCONV1D':

                model = nn_models.NORMCONV1D(n_timesteps, n_features, n_outputs, nkerns, filterSizes, poolSizes, activationConv,
                                            inputMLP, activationMLP, withBatchNormalization=True)

            elif modelname == 'CONV2D':

                model = nn_models.CONV2D(n_timesteps, n_features, n_outputs, nkerns, filterSizes, activationConv, inputMLP,
                                        activationMLP, withBatchNormalization=True)

            elif modelname == 'DEEPCONVLSTM':

                model = nn_models.DEEPCONVLSTM(n_timesteps, n_features, n_outputs, nkerns, filterSizes, poolSizes,
                                            activationConv, inputMLP, activationMLP)

            elif modelname == 'CONVLSTM_1D_HYBRID':
                trainx = trainx.reshape(
                    (trainx.shape[0], n_steps, n_length, n_features))
                testx = testx.reshape(
                    (testx.shape[0], n_steps, n_length, n_features))
                valx = valx.reshape((valx.shape[0], n_steps, n_length, n_features))
                model = nn_models.CONVLSTM_1D_NONHYBRID(
                    n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, inputMLP)

            elif modelname == 'CONVLSTM_1D':
                trainx = trainx.reshape(
                    (trainx.shape[0], n_steps, n_length, n_features))
                testx = testx.reshape(
                    (testx.shape[0], n_steps, n_length, n_features))
                valx = valx.reshape((valx.shape[0], n_steps, n_length, n_features))
                model = nn_models.CONVLSTM_1D(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, inputMLP,
                                            activationMLP, activationConv)

            elif modelname == 'CONVLSTM_2D_HYBRID':
                trainx = trainx.reshape(
                    (trainx.shape[0], n_steps, 1, n_length, n_features))
                testx = testx.reshape(
                    (testx.shape[0], n_steps, 1, n_length, n_features))
                valx = valx.reshape(
                    (valx.shape[0], n_steps, 1, n_length, n_features))
                model = nn_models.CONVLSTM_2D_NONHYBRID(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns,
                                                        activationConv, activationMLP, inputMLP)

            elif modelname == 'CONVLSTM_2D':
                trainx = trainx.reshape(
                    (trainx.shape[0], n_steps, 1, n_length, n_features))
                testx = testx.reshape(
                    (testx.shape[0], n_steps, 1, n_length, n_features))
                valx = valx.reshape(
                    (valx.shape[0], n_steps, 1, n_length, n_features))
                model = nn_models.CONVLSTM_2D(n_steps, n_length, n_features, n_outputs, filterSizes, nkerns, activationConv,
                                            activationMLP, inputMLP)

                # Compiling The model
            model.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

            print("Model created")
            print(model.summary())

            # Fitting the model With the system.
            print("Fit model:")

            for i in range(num_repeats):
                print(f"Experiment {i+1}/{num_repeats}")

                # Fit the model
                model.fit(trainx, trainy, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, validation_data=(valx, valy))
                # model.fit(trainx_f, trainy, batch_size=batch_size, epochs=epochs, verbose=verbose)

                # Evaluating the model
                print("Evaluate model: ")

                # Evaluate model
                loss, accuracy = model.evaluate(testx, testy, verbose=verbose)
                testing_pred = model.predict(testx)
                testing_pred = testing_pred.argmax(axis=-1)
                true_labels = testy.argmax(axis=-1)
                f1scores_per_class = (
                    f1_score(true_labels, testing_pred, average=None))
                average_fscore_macro = (
                    f1_score(true_labels, testing_pred, average="macro"))
                average_fscore_weighted = (
                    f1_score(true_labels, testing_pred, average="weighted"))
                print('Average F1 Score per class:', f1scores_per_class)
                print('Average_Macro F1 Score of the Model:', average_fscore_macro)
                print('Average_Weighted F1 score of the Model:', average_fscore_weighted)
                print('Accuracy of the Model:', accuracy)

                # time_elapsed = datetime.now() - start_time

                saveResultsCSV(method, factor, modelname, epochs, batch_size, accuracy,
                            average_fscore_macro, average_fscore_weighted, "", "")

            tf.keras.backend.clear_session()

            del model, trainx, trainy, valx, valy, testx, testy

            gc.collect()

    # -------------------------------------------------------------------------------------------------------------
        createandevaluate(method, factor, trainx, testx, valx, trainy, testy, valy,  n_timesteps, n_features, n_outputs, n_steps, n_length, nkerns, filterSizes,
                    poolSizes, activationConv, inputMLP, activationMLP, verbose, epochs, batch_size,
                    start_time=datetime.now())
