def slided_numpy_array(data):
    import numpy as np
    # This function will generate the 3D Data for the DEEP CNN model
    # Input is a 2D array where the last column contains the labels information
    # x = data.to_numpy()
    def get_strides(a, L, ov):
        out = []

        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i : i + L, :])
        return np.array(out)

    L = 128
    ov = 0

    # print('After Overlapping')
    x = get_strides(data, L, ov)
    # print(x.shape)

    segment_idx = 0  # Index for the segment dimension
    nb_segments, nb_timestamps, nb_columns = x.shape
    data_to_save = np.zeros(
        (nb_segments, nb_timestamps, nb_columns - 1), dtype=np.float32
    )
    labels_to_save = np.zeros(nb_segments, dtype=int)

    for i in range(0, nb_segments):
        data_3D = x[i][:][:]
        data_to_save[i] = data_3D[:, :-1]
        labels = data_3D[:, -1]
        labels = labels.astype("int")  # Convert labels to int to avoid typing issues
        values, counts = np.unique(labels, return_counts=True)
        labels_to_save[i] = values[np.argmax(counts)]

    return data_to_save, labels_to_save
