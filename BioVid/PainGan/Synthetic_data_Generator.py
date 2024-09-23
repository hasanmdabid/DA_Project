
#pylint: disable-all
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from joblib import load

dataset_name = input("painmonit or biovid?")
def GAN_generator(n, dataset_name):
    ##########################################################
    # Now, let us load the generator model and generate images
    # Lod the trained model and generate a few images
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
    
    if dataset_name == 'painmonit':
        n_classes = 1             # All the classes except the majority class(0)
        label_range = int(n_samples +1)
        print("Print label range:", label_range)
        print("Type of Label range:", type(label_range))
        # Load GAN model for pain-OPPO model
        model_pain = load_model('/home/abidhasan/Documents/PAINGAN/results/trained_model/CGAN_PAIN_painmonit_2000epochs_64Batch.h5')
        scaler = load('/home/abidhasan/Documents/PAINGAN/painmonit_minmax_scaler.bin')
        labels_pain = asarray(
        [x for _ in range(1, label_range) for x in range (1, 2)])  # Dimension of Labels should be same as N_samples. *****
    elif dataset_name== 'biovid':
        n_classes = 2             # All the class becasuse the datset is balance.
        label_range = int((n_samples//2) +1)
        print("Print label range:", label_range)
        print("Type of Label range:", type(label_range))
        # Load GAN model for pain-OPPO model
        model_pain = load_model('/home/abidhasan/Documents/PAINGAN/results/trained_model/CGAN_PAIN_biovid_2000epochs_64Batch.h5')
        scaler = load('/home/abidhasan/Documents/PAINGAN/biovid_minmax_scaler.bin')
        labels_pain = asarray(
        [x for _ in range(1, label_range) for x in range (0, 2)])  # Dimension of Labels should be same as N_samples. *****
    
    # ----------------------------------------------pain OPPO-----------------------------------------------------------------
    # Load GAN model for pain-OPPO model
    
    latent_points_pain, _ = generate_latent_points(latent_dim, n_samples, n_classes)  # Input (Latent Point Dimension, n_Samples) .
    # specify labels - generate 10 sets of labels each gping from 0 to 9
    
    print("Shape of pain Latent point:", latent_points_pain.shape)
    print("Shape of pain Labels:", labels_pain.shape)
    print(labels_pain)

    # Generate pain Data
    X_pain = model_pain.predict([latent_points_pain, labels_pain])
    print('Shape of pain generated data', X_pain.shape)

    max_val_sacled = np.max(X_pain)
    min_val_scaled = np.min(X_pain)
    print('Maximume_value Before rescaled:', max_val_sacled)
    print('Minimume_value Before rescaled:', min_val_scaled)

    nr_samples, nr_rows, nr_columns, nr_channels = X_pain.shape
    # Rescale from [-1, 1] by using the MinMax scaler inverse transform
    X_pain = X_pain.reshape(nr_samples * nr_rows, nr_columns)
    # Rescale from [-1, 1] by using the MinMax scaler inverse transform
    X_pain = scaler.inverse_transform(X_pain)
    X_pain = X_pain.reshape(nr_samples, nr_rows, nr_columns)   # Rehping into [Nr_Samples, Nr_rows, Nr_Columns]
    print('After rescalling and reshape of pain generated data', X_pain.shape)

    max_val = np.max(X_pain)
    min_val = np.min(X_pain)
    print('Maximume_value after scaled:', max_val)
    print('Minimume_value after scaled:', min_val)

    # -----------------------------------------------Checking the Labels ratio-----------------------------------------------

    import collections
    unique, counts = np.unique(labels_pain, return_counts=True)
    counter = collections.Counter(labels_pain)
    print(counter)
    plt.bar(counter.keys(), counter.values())
    plt.savefig(f'/home/abidhasan/Documents/PAINGAN/picture/{dataset_name}_bar_plot_of_{factor}_Synthetic_Data.png', dpi=400)

    # ---------------------------------------------Checking the Maximume and Minimume value----------------------------------
    return X_pain, labels_pain

if dataset_name == 'painmonit':
    factors = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4]
    pain_data_samples = 252  # Generatin
elif dataset_name == 'biovid':
    factors = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4]
    pain_data_samples = 3480  # Generatin



for factor in factors:
    n = pain_data_samples*factor
    X_synthetic, y_synthetic = GAN_generator(n, dataset_name)
    np.save(f'/home/abidhasan/Documents/PAINGAN/Data/cGAN_Generated_data/{dataset_name}/{factor}_X', X_synthetic)
    np.save(f'/home/abidhasan/Documents/PAINGAN/Data/cGAN_Generated_data/{dataset_name}/{factor}_y', y_synthetic)
