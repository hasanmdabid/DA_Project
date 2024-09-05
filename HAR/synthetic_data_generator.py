# pylint: disable-all

from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt

def GAN_generator(factor, nr_samples_HAR, nr_classes=6):
    ##########################################################
    # Now, let us load the generator model and generate images
    # Load the trained model and generate a few images
    # ---------------------------------------------------------------------------------------------------------------------
    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, nr_samples, nr_classes):
        # generate points in the latent space
        x_input = randn(latent_dim * nr_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(nr_samples, latent_dim)
        # generate labels
        labels = randint(0, nr_classes, nr_samples)
        return [z_input, labels]

    # **** The number of samples value should be the factor of nr_classes
    nr_samples = ((nr_classes * int(factor * nr_samples_HAR)))
    latent_dim = 100  # 
    # * All the classes Becasue the data set is banalced
    label_range = int(nr_samples_HAR*factor)+1
    # ----------------------------------------------HAR OPPO-----------------------------------------------------------------
    # Load GAN model for HAR-OPPO model
    model_har = load_model('/home/abidhasan/Documents/HAR/cGAN_HAR_200_epochs_64_Batch.h5')

    # Input (Latent Point Dimension, nr_samples) .
    latent_points_har, _ = generate_latent_points(latent_dim, nr_samples, nr_classes)
    # specify labels - generate 10 sets of labels each gaping from 0 to 9
    labels_har = asarray(
        [x for _ in range(1, label_range) for x in range(1, nr_classes+1)])  # Dimension of Labels should be same as nr_samples. *****
    print("Shape of Har Latent point:", latent_points_har.shape)
    print("Shape of Har Labels:", labels_har.shape)
    print(labels_har)
    from joblib import load
    scaler = load('/home/abidhasan/Documents/HAR/minmax_scaler.bin')

    # Generate Har Data


    X_har = model_har.predict([latent_points_har, labels_har])
    print('Shape of HAR generated data', X_har.shape)

    max_val_sacled = np.max(X_har)
    min_val_scaled = np.min(X_har)
    print('Maximume_value Before rescaled:', max_val_sacled)
    print('Minimume_value Before rescaled:', min_val_scaled)

    nr_samples, nr_rows, nr_columns, nr_channels = X_har.shape
    # Converting X_har to 3D to 2D 
    X_har = X_har.reshape(nr_samples * nr_rows, nr_columns)
    # Rescale from [-1, 1] by using the MinMax scaler inverse transform
    X_har = scaler.inverse_transform(X_har)
    # Rehping into [Nr_Samples, Nr_rows, Nr_Columns]
    X_har = X_har.reshape(nr_samples, nr_rows, nr_columns)
    print('After rescalling and reshape of HAR generated data', X_har.shape)

    max_val = np.max(X_har)
    min_val = np.min(X_har)
    print('Maximume_value after scaled:', max_val)
    print('Minimume_value after scaled:', min_val)

    # -----------------------------------------------Checking the Labels ratio-----------------------------------------------

    import collections
    unique, counts = np.unique(labels_har, return_counts=True)
    counter = collections.Counter(labels_har)
    print(counter)
    plt.bar(counter.keys(), counter.values())
    plt.savefig('Bar_plot_of_Class_Distribution_of_Synthetic_Data', dpi=400)

    # ---------------------------------------------Checking the Maximume and Minimume value----------------------------------
    return X_har, labels_har


print(__name__)
if __name__ == '__main__':
    try:
        X_har, labels_har = GAN_generator(factor, nr_classes=6)
    except TypeError:
        print("Something went wrong")
    except NameError:
        print("Factor is not a Defined")
