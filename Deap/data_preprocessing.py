def data_pre():
    import pickle
    import matplotlib
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # ----------------------------------------------------------Loading the Data---------------------------------------------
    # The EEG and peripheral physiological signals of 32 participants were recorded as each watched 40 music videos.
    # Participants rated each video in terms of the levels of arousal, valence, dominance, and liking. The data was
    # downsampled (to 128Hz), preprocessed and segmented in pickled Python formats.
    # Function to load data from each participant file
    # Function to load data from each participant file
    def read_eeg_signal_from_file(filename):
        x = pickle._Unpickler(open(filename, 'rb'))
        x.encoding = 'latin1'
        p = x.load()
        return p

    # Load only 22/32 participants with frontal videos recorded
    files = []
    for n in range(1, 33):
        s = ''
        if n < 10:
            s += '0'
        s += str(n)
        files.append(s)
    print(files)

    # Each participant file contains two arrays: a "data" array (40 trials x 40 channels x 8064 data) and a "label" array
    # (40 trials x 4 subjective ratings: valence, arousal, dominance, liking). We combine the data files into 2 new arrays
    # with 1280 trials for 32 participants (participants with frontal videos recorded)
    # 32x40 = 1280 trials for 22 participants
    labels = []
    data = []

    for i in files:
        filename = "/home/abidhasan/Documents/DA_Project/Deap/data/s" + i + ".dat"
        trial = read_eeg_signal_from_file(filename)
        labels.append(trial['labels'])
        data.append(trial['data'])

    # Re-shape arrays into desired shapes
    labels = np.array(labels)
    print(labels.shape)
    labels = labels.flatten()
    labels = labels.reshape(1280, 4)

    data = np.array(data)

    data = data.flatten()
    data = data.reshape(1280, 40, 8064)

    # Double-check the new arrays
    print("Raw Labels: ", labels.shape)  # trial x label
    print("RawData: ", data.shape)  # trial x channel x data

    # --------------------------------------Explore and pre-process data-----------------------------------------------------
    # Labels
    # Get Valence and Arousal ratings
    # Valence describes the extent to which an emotion is positive or negative, whereas Arousal refers to its intensity,i.e.,
    # the strength of the associated emotional state.

    # Only extract Valence and Arousal ratings
    df_label_ratings = pd.DataFrame(
        {'Valence': labels[:, 0], 'Arousal': labels[:, 1]})

    # Check positive/negative cases
    # The combinations of Valence and Arousal can be converted to emotional states: High Arousal Positive Valence
    # (Excited, Happy), Low Arousal Positive Valence (Calm, Relaxed), High Arousal Negative Valence (Angry, Nervous),
    # and Low Arousal Negative Valence (Sad, Bored).

    # High Arousal Positive Valence dataset
    df_hahv = df_label_ratings[
        (df_label_ratings['Valence'] >= np.median(labels[:, 0])) & (
            df_label_ratings['Arousal'] >= np.median(labels[:, 1]))]
    # Low Arousal Positive Valence dataset
    df_lahv = df_label_ratings[
        (df_label_ratings['Valence'] >= np.median(labels[:, 0])) & (
            df_label_ratings['Arousal'] < np.median(labels[:, 1]))]
    # High Arousal Negative Valence dataset
    df_halv = df_label_ratings[
        (df_label_ratings['Valence'] < np.median(labels[:, 0])) & (
            df_label_ratings['Arousal'] >= np.median(labels[:, 1]))]
    # Low Arousal Negative Valence dataset
    df_lalv = df_label_ratings[
        (df_label_ratings['Valence'] < np.median(labels[:, 0])) & (
            df_label_ratings['Arousal'] < np.median(labels[:, 1]))]

    # Valence and Arousal ratings between groups

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].set_title("Valence")
    axs[0].set_ylim(1, 9)
    axs[0].boxplot([df_hahv['Valence'], df_lahv['Valence'], df_halv['Valence'], df_lalv['Valence']],
                   labels=['HAHV', 'LAHV', 'HALV', 'LALV'])

    axs[1].set_title("Arousal")
    axs[1].set_ylim(1, 9)
    axs[1].boxplot([df_hahv['Arousal'], df_lahv['Arousal'], df_halv['Arousal'], df_lalv['Arousal']],
                   labels=['HAHV', 'LAHV', 'HALV', 'LALV'])

    plt.savefig('Valance and Arousal between groups.png')

    # Valence and Arousal ratings per group
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].set_title("HAHV")
    axs[0, 0].set_ylim(1, 9)
    axs[0, 0].boxplot([df_hahv['Valence'], df_hahv['Arousal']],
                      labels=['Valence', 'Arousal'])

    axs[0, 1].set_title("LAHV")
    axs[0, 1].set_ylim(1, 9)
    axs[0, 1].boxplot([df_lahv['Valence'], df_lahv['Arousal']],
                      labels=['Valence', 'Arousal'])

    axs[1, 0].set_title("HALV")
    axs[1, 0].set_ylim(1, 9)
    axs[1, 0].boxplot([df_halv['Valence'], df_halv['Arousal']],
                      labels=['Valence', 'Arousal'])

    axs[1, 1].set_title("LALV")
    axs[1, 1].set_ylim(1, 9)
    axs[1, 1].boxplot([df_lalv['Valence'], df_lalv['Arousal']],
                      labels=['Valence', 'Arousal'])
    plt.savefig('Valence and arousal rating per group.png')

    # ----------------One hot encoding----------------------------

    # Function to check if each trial has positive or negative valence
    def positive_valence(trial):
        return 1 if labels[trial, 0] >= np.median(labels[:, 0]) else 0

    # Function to check if each trial has high or low arousal
    def high_arousal(trial):
        return 1 if labels[trial, 1] >= np.median(labels[:, 1]) else 0

    # Convert all ratings to boolean values
    labels_encoded = []
    for i in range(len(labels)):
        labels_encoded.append([positive_valence(i), high_arousal(i)])
    labels_encoded = np.reshape(labels_encoded, (1280, 2))
    df_labels = pd.DataFrame(data=labels_encoded, columns=[
                             "Positive Valence", "High Arousal"])

    # print('Details of Labels', df_labels.describe())

    # Dataset with only Valence column
    df_valence = df_labels['Positive Valence']
    #print('Printing the 1st 10 valance labels:\n', df_valence.head(10))
    print('Counting the number of occurance of valance labels\n', df_valence.value_counts())
    #print('Shape of Valance labels:', df_valence.shape)

    # Dataset with only Arousal column
    df_arousal = df_labels['High Arousal']
    #print('Printing the 1st 10 Arousal labels:\n', df_arousal.head(10))
    print('Counting the number of occurances of Arousal labels\n',df_arousal.value_counts())
    #print('Shape of Arousal labels:', df_arousal.shape)

    # Now we will convert the label to 32*40 = 1280 (trials) * 8064 (data_samples) = 10321920
    """
    x = np.array([[1,2],[3,4], [5,6]])
    print(x.shape)
    x = np.repeat(x, 5, axis=0)
    print(x.shape)
    print(x)
    """
    valence = np.repeat(np.array(df_valence), 8064, axis=0)  # Converting the labels into total number of time stamp
    # Converting the labels into total number of time stamp
    arousal = np.repeat(np.array(df_arousal), 8064, axis=0)

    print('Now the shape of total Valance labels:', valence.shape)
    print('Now the shape of total Arousal labels:', arousal.shape)

    # -----------------------------------------------------------EEG data----------------------------------------------------
    # -------------------------------------------Separate EEG and non-EEG data---------------------------------------------
    # The dataset includes 32 EEG channels and 8 peripheral physiological channels. The peripheral signals include:
    # electrooculogram (EOG), electromyograms (EMG) of Zygomaticus and Trapezius muscles, GSR, respiration amplitude,
    # blood volume by plethysmograph, skin temperature.
    eeg_channels = np.array(
        ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2",
         "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"])
    peripheral_channels = np.array(
        ["hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Respiration belt", "Plethysmograph", "Temperature"])

    eeg_data = []
    for i in range(len(data)):
        for j in range(len(eeg_channels)):
            eeg_data.append(data[i, j])
    eeg_data = np.reshape(
        eeg_data, (len(data), len(eeg_channels), len(data[0, 0])))
    #print('Shape of EEG data', eeg_data.shape)

    peripheral_data = []
    for i in range(len(data)):
        for j in range(32, len(data[0])):
            peripheral_data.append(data[i, j])
    peripheral_data = np.reshape(peripheral_data, (len(
        data), len(peripheral_channels), len(data[0, 0])))
    #print('Shappe of Peripheral data', peripheral_data.shape)

    # Converting the data into (Samples, row, column) form.

    eeg_data = eeg_data.transpose(0, 2, 1)
    peripheral_data = peripheral_data.transpose(0, 2, 1)

    #print('Eeg data in samples, row, column formate:', eeg_data.shape)
    #print('Peripheral data in samples, row, column formate:', peripheral_data.shape)

    # --------------------------------------Sliding Window approach----------------------------------------------------------

    # Applying the Sliding window approach for 5 Sec Window. The sampling frequency is 128Hz (128 samples per sec) for DEAP
    # data set. SO we will select the time stamp 5 sec (640 samples per segment).
    # 1st we will convert the 3D data into 2D data.

    eeg_data = eeg_data.reshape(eeg_data.shape[0] * eeg_data.shape[1], eeg_data.shape[2])
    peripheral_data = peripheral_data.reshape(peripheral_data.shape[0] * peripheral_data.shape[1], peripheral_data.shape[2])
    #print('EEG data in 2D', eeg_data.shape)
    #print('Peripheral data in 2D', peripheral_data.shape)

    combined_data = np.hstack((eeg_data, peripheral_data))

    print('Combined data:', combined_data.shape)

    # ------------------------------------------Data saling-----------------------------------------------------------
    trans = StandardScaler()
    combined_data = trans.fit_transform(combined_data)
    eeg_data = trans.fit_transform(eeg_data)
    peripheral_data = trans.fit_transform(peripheral_data)

    # print('Detailed of Data set after normalization:\n', combined_data.describe())

    eeg_valence = np.append(eeg_data, valence.reshape(valence.shape[0], 1), axis=1)
    eeg_arousal = np.append(eeg_data, arousal.reshape(arousal.shape[0], 1), axis=1)
    #print('After the concatination shape of eeg and valance', eeg_valence.shape)
    #print('After the concatination shape of eeg and arousal', eeg_arousal.shape)

    combined_data_valence = np.append(combined_data, valence.reshape(valence.shape[0], 1), axis=1)
    combined_data_arousal = np.append(combined_data, arousal.reshape(arousal.shape[0], 1), axis=1)

    print('After the concatination shape of combined and valance', combined_data_valence.shape)
    print('After the concatination shape of combined and arousal', combined_data_arousal.shape)

    # This function will generate the slided Windowed Data
    def slided_numpy_array(data):
        # x = data.to_numpy()
        def get_strides(a, L, ov):
            out = []

            for i in range(0, a.shape[0] - L + 1, L - ov):
                out.append(a[i:i + L, :])
            return np.array(out)

        L = 128
        ov = 0

        # print('After Overlapping')
        x = get_strides(data, L, ov)
        # print(x.shape)

        segment_idx = 0  # Index for the segment dimension
        nb_segments, nb_timestamps, nb_columns = x.shape
        data_to_save = np.zeros(
            (nb_segments, nb_timestamps, nb_columns - 1), dtype=np.float32)
        labels_to_save = np.zeros(nb_segments, dtype=int)

        for i in range(0, nb_segments):
            labels = x[i][:][:]
            data_to_save[i] = labels[:, :-1]
            labels = x[i][:][:]
            labels = labels[:, -1]
            # Convert labels to int to avoid typing issues
            labels = labels.astype('int')
            values, counts = np.unique(labels, return_counts=True)
            labels_to_save[i] = values[np.argmax(counts)]

        return data_to_save, labels_to_save

    eeg_valence_slided, valence_slided = slided_numpy_array(eeg_valence)
    eeg_arousal_slided, arousal_slided = slided_numpy_array(eeg_arousal)
    combined_data_valence_slided, combined_valence_slided = slided_numpy_array(combined_data_valence)
    combined_data_arousal_slided, combined_arousal_slided = slided_numpy_array(combined_data_arousal)

    #print('After sliding-window 3D data', combined_data_valence_slided.shape)
    #print('After sliding-window labels', combined_valence_slided.shape)
    unique_valence, counts_valence = np.unique(combined_valence_slided, return_counts=True)
    #print(np.asarray((unique_valence, counts_valence)).T)

    #print('After sliding-window 3D data', combined_data_arousal_slided.shape)
    #print('After sliding-window labels', combined_arousal_slided.shape)
    unique_arousal, counts_arousal = np.unique(arousal_slided, return_counts=True)
    #print(np.asarray((unique_arousal, counts_arousal)).T)

    return eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided
