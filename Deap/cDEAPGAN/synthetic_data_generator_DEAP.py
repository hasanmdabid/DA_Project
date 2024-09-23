# pylint: disable-all
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import os

def GAN_generator(n, label, factor):
    ##########################################################
    # Now, let us load the generator model and generate signals
    # Lod the trained model and generate signals
    # ---------------------------------------------------------------------------------------------------------------------
    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples, n_classes):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        # generate labels
        labels = randint(0, n_classes, n_samples)
        return [z_input, labels]

    n_samples = int(n)                         #**** The number of samples value should be the factor of N_classes
    latent_dim = 100                           #*
    n_classes = 2                            #* All the classes except the Minority class (0).
    label_range = int((n_samples//2) +1)
    print("Print label range:", label_range)
    print("Type of Label range:", type(label_range))
    # ----------------------------------------------DEAP OPPO-----------------------------------------------------------------
    # Load GAN model for DEAP-OPPO model
    model_DEAP = load_model(
        '/home/abidhasan/Documents/DA_Project/Deap/cDEAPGAN/results/trained_model/CGAN_DEAP_'+label+'_500epochs_64Batch.h5')
    latent_points_DEAP, _ = generate_latent_points(latent_dim, n_samples, n_classes)  # Input (Latent Point Dimension, n_Samples) .

    labels_DEAP = asarray(
        [x for _ in range(1, label_range) for x in range (0, 2)])  # Dimension of Labels should be same as N_samples. *****
    print("Shape of DEAP Latent point:", latent_points_DEAP.shape)
    print("Shape of DEAP Labels:", labels_DEAP.shape)
    print(labels_DEAP)
    from joblib import load
    scaler = load(
        '/home/abidhasan/Documents/DA_Project/Deap/cDEAPGAN/'+label+'_minmax_scaler.bin')

    # Generate DEAP Data
    X_DEAP = model_DEAP.predict([latent_points_DEAP, labels_DEAP])
    print('Shape of DEAP generated data', X_DEAP.shape)

    max_val_sacled = np.max(X_DEAP)
    min_val_scaled = np.min(X_DEAP)
    print('Maximume_value Before rescaled:', max_val_sacled)
    print('Minimume_value Before rescaled:', min_val_scaled)

    nr_samples, nr_rows, nr_columns, nr_channels = X_DEAP.shape
    # Rescale from [-1, 1] by using the MinMax scaler inverse transform
    X_DEAP = X_DEAP.reshape(nr_samples * nr_rows, nr_columns)
    # Rescale from [-1, 1] by using the MinMax scaler inverse transform
    X_DEAP = scaler.inverse_transform(X_DEAP)
    X_DEAP = X_DEAP.reshape(nr_samples, nr_rows, nr_columns)   # Rehping into [Nr_Samples, Nr_rows, Nr_Columns]
    print('After rescalling and reshape of DEAP generated data', X_DEAP.shape)

    max_val = np.max(X_DEAP)
    min_val = np.min(X_DEAP)
    print('Maximume_value after scaled:', max_val)
    print('Minimume_value after scaled:', min_val)

    # -----------------------------------------------Checking the Labels ratio-----------------------------------------------

    import collections
    unique, counts = np.unique(labels_DEAP, return_counts=True)
    counter = collections.Counter(labels_DEAP)
    print(counter)
    plt.bar(counter.keys(), counter.values())
    plt.savefig(f'/home/abidhasan/Documents/DA_Project/Deap/cDEAPGAN/figures/' +
                label+'_bar_plot_of_{factor}_Synthetic_Data.png', dpi=400)

    # ---------------------------------------------Checking the Maximume and Minimume value----------------------------------
    return X_DEAP, labels_DEAP


factors = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4]
Deap_data_samples = 80640
labels = ['arousal', 'valance']

for label in labels:
    for factor in factors:
        n = Deap_data_samples * factor
        X_synthetic, y_synthetic = GAN_generator(n, label, factor)

        # Define the paths
        dir_path = f'/home/abidhasan/Documents/DA_Project/Deap/Data/cGAN_generated_data/{label}'
        X_path = os.path.join(dir_path, f'{factor}_X.npy')
        y_path = os.path.join(dir_path, f'{factor}_y.npy')

        # Create the directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save the files
        np.save(X_path, X_synthetic)
        np.save(y_path, y_synthetic)

        print(f'Saved {X_path} and {y_path}')
