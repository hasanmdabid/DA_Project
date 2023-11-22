# This function will generate the slided Windowed Data
import numpy as np
def slided_numpy_array(data):
    # x = data.to_numpy()
    def get_strides(a, L, ov):
        out = []

        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])
        return np.array(out)

    L = 128 # We choose 128 sec time window. 
    ov = 0

    # print('After Overlapping')
    x = get_strides(data, L, ov)
    # print(x.shape)

    segment_idx = 0  # Index for the segment dimension
    nb_segments, nb_timestamps, nb_columns = x.shape
    data_to_save = np.zeros((nb_segments, nb_timestamps, nb_columns - 1), dtype=np.float32)
    labels_to_save = np.zeros(nb_segments, dtype=int)

    for i in range(0, nb_segments):
        data = x[i][:][:]
        data_to_save[i] = data[:, :-1]
        labels = data[:, -1]
        # Convert labels to int to avoid typing issues
        labels = labels.astype('int')
        values, counts = np.unique(labels, return_counts=True)
        labels_to_save[i] = values[np.argmax(counts)]

    return data_to_save, labels_to_save

print(__name__)
if __name__ == '__main__':
    try:
        data_to_save, labels_to_save = slided_numpy_array(data)
    except TypeError:
        print("Something went wrong")
    except NameError:
        print("Data is not Defined")