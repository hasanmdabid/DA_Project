import numpy as np
from sklearn.utils import resample
import math



def slided_numpy_array(data, L=128, ov = 0 ):
    import numpy as np
    # This function will generate the 3D Data for the DEEP CNN model
    # Input is a 2D array where the last column contains the labels information
    # x = data.to_numpy()
    def get_strides(a, L, ov):
        out = []

        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i : i + L, :])
        return np.array(out)

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



# Assuming you have 3D data x and y
x = np.random.rand(4, 4, 4)  # Replace with your actual data
y = np.random.randint(0, 2, size=(4, 4, 4))  # Example labels, replace with your actual labels
print(x.shape, y.shape)
# Reshape x and y to 2D
x_2d = x.reshape(x.shape[0]*x.shape[1], x.shape[2])  #Converting data from 3D to 2D
y_2d = x.reshape(y.shape[0]*x.shape[1], x.shape[2])  #Converting data from 3D to 2D
print(x_2d.shape, y_2d.shape)
# Combine x and y into a single dataset
dataset = np.hstack((x_2d, y_2d))
print(dataset.shape)
# Randomly select 20% of the dataset
np.random.shuffle(dataset)
split_index = int(0.2 * dataset.shape[0])
sub_dataset = dataset[:split_index]

# Reshape sub_dataset back to 3D arrays
x_subset = sub_dataset[:, :-1].reshape(x.shape)
y_subset = sub_dataset[:, -1].reshape(y.shape)

# Check the shapes of x_subset and y_subset
print("x_subset shape:", x_subset.shape)
print("y_subset shape:", y_subset.shape)


Aug_frac = 0.2  # Select the values of the augmatation fraction.   
n_ecp_samples = math.ceil(dataset.shape[0]*Aug_frac) 
print('Shape of Number of Expected samples:', n_ecp_samples)

#---------------------------------------------------------------- Performing the Downsample With SKlearn----------------------------------------------------------------
# We applied resample() method from the sklearn.utils module for downsampling, The replace = True attribute performs random resampling with replacement. The n_samples attribute 
# defines the number of records you want to select from the original records. We have set the value of this attribute to the number of records in the spam dataset so the two sets will be balanced.
Aug_downsample = resample(dataset, replace=True, n_samples=n_ecp_samples, random_state=42)

print(Aug_downsample.shape)

#---------------------------------------------------Converting the Labels into 1D array------------------------------------------------
# This part of code will 1st selec the Number of samples  will 
Aug_frac_x, Aug_frac_y = slided_numpy_array(Aug_downsample)
print('Shape of training Augmented data with fractional amount:',Aug_frac_x.shape)
print('Shape of training Augmented data with fractional amount:',Aug_frac_y.shape)

x_train = np.concatenate((x,Aug_frac_x), axis=0)
y_train = np.concatenate((y,Aug_frac_y), axis=0)
print('X_train data shape after fractional Augmentation:', x_train.shape)
print('Y_train data shape after fractional Augmentation:', y_train.shape)
