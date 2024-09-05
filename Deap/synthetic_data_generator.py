# pylint: disable-all

def GAN_generator():
    ##########################################################
    # Now, let us load the generator model and generate 
    # Lod the trained model and generate a few 
    from numpy import asarray
    from numpy.random import randn
    from numpy.random import randint
    from keras.models import load_model
    import numpy as np
    from matplotlib import pyplot as plt

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

    n_samples = 200                          #**** The number of samples value should be the factor of N_classes
    latent_dim = 100                           #*
    n_classes = 2                            #* All the classes except the Minority class (0).

    # ----------------------------------------------HAR OPPO-----------------------------------------------------------------
    # Load GAN model for HAR-OPPO model
    model_har = load_model('/home/abidhasan/Documents/DA_Project/Deap/CGAN_DEAP_valance_500epochs_64Batch.h5')
    latent_points_har, _ = generate_latent_points(latent_dim, n_samples, n_classes)  # Input (Latent Point Dimension, n_Samples) .
    # specify labels - generate 10 sets of labels each gping from 0 to 9
    labels_har = asarray(
        [x for _ in range(1, 101) for x in range (0, 2)])  # Dimension of Labels should be same as N_samples. *****
    print("Shape of Har Latent point:", latent_points_har.shape)
    print("Shape of Har Labels:", labels_har.shape)
    print(labels_har)
    from joblib import load
    scaler = load('/home/abidhasan/Documents/DA_Project/Deap/valance_minmax_scaler.bin')

    # Generate Har Data
    X_har = model_har.predict([latent_points_har, labels_har])
    print('Shape of generated data', X_har.shape)

    max_val_sacled = np.max(X_har)
    min_val_scaled = np.min(X_har)
    print('Maximume_value Before rescaled:', max_val_sacled)
    print('Minimume_value Before rescaled:', min_val_scaled)

    nr_samples, nr_rows, nr_columns, nr_channels = X_har.shape
    # Rescale from [-1, 1] by using the MinMax scaler inverse transform
    X_har = X_har.reshape(nr_samples * nr_rows, nr_columns)
    # Rescale from [-1, 1] by using the MinMax scaler inverse transform
    X_har = scaler.inverse_transform(X_har)
    X_har = X_har.reshape(nr_samples, nr_rows, nr_columns)   # Rehping into [Nr_Samples, Nr_rows, Nr_Columns]
    print('After rescalling and reshape of  generated data', X_har.shape)

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