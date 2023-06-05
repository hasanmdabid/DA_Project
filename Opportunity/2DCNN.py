def model(trainx, trainy, testx, testy):
    # This projects will highlight the comparative analysis of GANS performance for both univariate and multivariate dataset
    # ## 1st step will be to design an CNN model for the opportunity dataset and check the performance of the model without
    # data augmentation.
    #    ### 1. Preprocessing the dataset
    #    ### 2. segmentation of the dataset
    # ## 2nd step will be to use existing augmentation method to generate fabricated data and see the performance.
    # ## 3rd step will be to formulate a unique evaluation matrix for the Augmentation methods (GANS)

    from keras.models import Sequential
    from keras.layers import Dense, Flatten
    from keras.layers import Dropout
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.utils import to_categorical
    from keras.layers import BatchNormalization
    from sklearn.metrics import f1_score

    ################################################################################################
    # The output data is defined as an integer for the class number. We must one hot encode these class
    # integers so that the data is suitable for fitting a neural network multi-class classification model.
    # We can do this by calling the to_categorical() Keras function.
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainx.shape, trainy.shape, testx.shape, testy.shape)

    n_timesteps, n_features, n_outputs = trainx.shape[1], trainx.shape[2], trainy.shape[1]

    ######################################################################################################

    # Now the data is ready to be used in the 1D CNN model

    ##################################################################################################

    model = Sequential()
    model.add(BatchNormalization(input_shape=(n_timesteps, n_features, 1)))

    # 1st convolutional + pooling + normalization layer
    # model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation='linear', input_shape=input_shape))
    model.add(Conv2D(filters=50, kernel_size=(11, 1), activation='relu', input_shape=(n_timesteps, n_features, 1)))
    model.add(MaxPooling2D(2, 1))

    # 2nd convolutional + pooling + normalization layer
    model.add(Conv2D(filters=40, kernel_size=(10, 1), activation='relu'))
    model.add(MaxPooling2D(3, 1))

    # 3rd block: convolutional + RELU + normalization
    model.add(Conv2D(filters=30, kernel_size=(6, 1), activation='relu'))
    model.add(MaxPooling2D(1, 1))

    # Fully-connected layer
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))

    # Softmax layer
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy',  # For autoencoders
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Model created")
    print(model.summary())

    print("Fit model:")
    verbose = 2
    epochs = 10
    batch_size = 100
    # Fit the model
    model.fit(trainx, trainy,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_data=(testx, testy))
    print("Evaluate model: ")

    # Evaluate model
    loss, accuracy = model.evaluate(testx, testy, verbose=verbose)
    testing_pred = model.predict(testx)
    testing_pred = testing_pred.argmax(axis=-1)
    true_labels = testy.argmax(axis=-1)

    f1scores_per_class = (f1_score(true_labels, testing_pred, average=None))
    average_fscore = (f1_score(true_labels, testing_pred, average="macro"))
    print('Average F1 Score per class:', f1scores_per_class)
    print('Average F1 Score of the Model:', average_fscore)
