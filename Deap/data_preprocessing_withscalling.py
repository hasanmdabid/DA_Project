def data_pre_pro_with_scalling():
    import pickle
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    def read_data(filename):
        x = pickle._Unpickler(open(filename, 'rb'))
        x.encoding = 'latin1'
        data = x.load()
        return data
    files = []
    for n in range(1, 33): 
        s = ''
        if n < 10:
            s += '0'
        s += str(n)
        files.append(s)
    # print(files)
    labels = []
    data = []
    for i in files: 
        fileph = "/home/abidhasan/Documents/DEAP/data/s" + i + ".dat"
        d = read_data(fileph)
        labels.append(d['labels'])
        data.append(d['data'])
    # print(labels)
    # # print(data)
    labels = np.array(labels)
    data = np.array(data)
    print('Raw labels', labels.shape)
    print('Raw data', data.shape)

    labels = labels.reshape(1280, 4)
    data = data.reshape(1280, 40, 8064)
    print(labels.shape)
    print(data.shape)
    #------ Converting the Data Into 2D------------------------------------------------------
    # # Converting the data into (Samples, row, column) form.
    data = data.transpose(0, 2, 1)                # data = (1280,8064,40)
    # Converting the data into 2D from 3D
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2]) # data = (1280*8064, 40)
    #------------------------Label creation-----------------------
    import pandas as pd
    df_label = pd.DataFrame({'Valence': labels[:,0], 'Arousal': labels[:,1], 
                        'Dominance': labels[:,2], 'Liking': labels[:,3]})

    label_name = ["valence","arousal","dominance","liking"]
    labels_valence = []
    labels_arousal = []
    labels_dominance = []
    labels_liking = []
    for la in labels:
        l = []
        if la[0]>5:
            labels_valence.append(1)
        else:
            labels_valence.append(0)
        if la[1]>5:
            labels_arousal.append(1)
        else:
            labels_arousal.append(0)
        if la[2]>5:
            labels_dominance.append(1)
        else:
            labels_dominance.append(0)
        if la[3]>6:
            labels_liking.append(1)
        else:
            labels_liking.append(0)
    valence = labels_valence 
    arousal = labels_arousal
    valence = np.repeat(np.array(valence), 8064, axis=0)  # Converting the labels into total number of time stamp
    arousal = np.repeat(np.array(arousal), 8064, axis=0)  # Converting the labels into total number of time stamp
    print('Shape of valance:', valence.shape)
    print('Shape of Arousal:', arousal.shape)
    # label_y = labels_dominance
    # label_y = labels_liking

    #------------------------------------------Data saling-----------------------------------------------------------
    trans = StandardScaler()
    minmax = MinMaxScaler()
    data = trans.fit_transform(data)

    data_valence = np.append(data, valence.reshape(valence.shape[0], 1), axis=1)
    data_arousal = np.append(data, arousal.reshape(arousal.shape[0], 1), axis=1)
    print('After the concatination shape of combined and valance', data_valence.shape)
    print('After the concatination shape of combined and arousal', data_arousal.shape)

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

    #eeg_valance_slided, valance_slided = slided_numpy_array(eeg_valence)
    #eeg_arousal_slided, arousal_slided = slided_numpy_array(eeg_arousal)
    data_valence_slided, valence_slided = slided_numpy_array(data_valence)
    data_arousal_slided, arousal_slided = slided_numpy_array(data_arousal)

    print('After sliding-window 3D data', data_valence_slided.shape)
    print('After sliding-window labels', valence_slided.shape)
    unique_valence, counts_valence = np.unique(valence_slided, return_counts=True)
    print(np.asarray((unique_valence, counts_valence)).T)

    print('After sliding-window 3D data', data_arousal_slided.shape)
    print('After sliding-window labels', arousal_slided.shape)
    unique_arousal, counts_arousal = np.unique(arousal_slided, return_counts=True)
    print(np.asarray((unique_arousal, counts_arousal)).T)
    
    return data_valence_slided, valence_slided, data_arousal_slided, arousal_slided

