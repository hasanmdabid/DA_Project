# pylint: disable-all
from joblib import dump
from pandas import read_csv, DataFrame
from numpy import dstack
import graphviz  # for showing model diagram
import matplotlib.pyplot as plt  # for data visualizationa
import matplotlib
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
# from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam , SGD
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose, MaxPool2D, BatchNormalization, UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from sklearn.preprocessing import MinMaxScaler
import os.path
from pathlib import Path
from datetime import datetime

# Importing Plotting Library
# from matplotlib import pyplot as plt
# from keras.utils import plot_model
# import graphviz  # for showing model diagram

import sys
import warnings

# Disabling the Python Warning.
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Data manipulation
import numpy as np  # for data manipulation

print('numpy: %s' % np.__version__)  # print version

# Visualization

print('matplotlib: %s' % matplotlib.__version__)  # print version

print('graphviz: %s' % graphviz.__version__)  # print version


# Disabling the Python Warning.
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# Load Sample Opportunity data
# load a single file as a numpy array


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]


def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test


def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_' +
                  group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_' +
                  group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_' +
                  group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# load the dataset, returns train and test X and y elements


def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group(
        'train', prefix + '/home/abidhasan/Documents/HAR/data/HARDataset/')
    print('Train 3D data:', trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group(
        'test', prefix + '/home/abidhasan/Documents/HAR/data/HARDataset/')
    print('Test 3D data:', testX.shape, testy.shape)
    return trainX, trainy, testX, testy

# Load the dataset, returns train and test X and y elements
trainX, trainy, testX, testy = load_dataset()
# define models

print("Maximume and minimume  value of trainX = {}, {}, and testX= {}, {}".format(np.max(trainX), np.min(trainX), np.max(testX), np.min(testX)))


########################################################################################################################
# Preparing the dataset to use for the CGAN
# Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
# Here I am Using The MinMaxScaler to normalize the data
# 1st I have reshaped the 3D trainX to 2D then perform the MinMaxsacler
# then I reshape it to 4D (Nr_samples, Nr_rows, Nr_columns, Nr_channels)
# Here channels = 1 (For Image channels = 3 for Colored image and 1 for the Gray scale Image)
# The dimension the of the trainX is 4d (Sample, Row, column, channels) and labels is Column Vector (Samples, 1)
########################################################################################################################

# Get the shape of the original 3D array
original_shape = trainX.shape

# Reshape the 3D array to 2D
trainx_reshaped = trainX.reshape(-1, original_shape[-1])

# Initialize the MinMaxScaler with the desired feature range
scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit and transform the reshaped array
trainx_scaled = scaler.fit_transform(trainx_reshaped)

dump(scaler, 'minmax_scaler.bin', compress=True)

# Reshape back to the original 3D shape
trainx_scaled = trainx_scaled.reshape(original_shape)
print("Maximume and minimume  value of trainx_scaled = {}, {}".format(
    np.max(trainx_scaled), np.min(trainx_scaled)))

print('Unique Indices in train set:', np.unique(trainy))
print('Unique indices in test  set:', np.unique(testy))
unique, counts = np.unique(trainy, return_counts=True)
print('Number of instances in train Y:')
print(np.asarray((unique, counts)).T)

# Add a new dimension for the number of channels
trainx_scaled = np.expand_dims(trainx_scaled, axis=-1)
dataset = [trainx_scaled, trainy]
print('Trainx_Scaled shape: ', trainx_scaled.shape)
print('Trainy: ', trainy.shape)
del trainX, trainx_reshaped

##################################################################################################################
# Define generator, discriminator, gan and other helper functions
# We will use functional way of defining model as we have multiple inputs;
# both images and corresponding labels.
##################################################################################################################

# define the standalone discriminator model
# Given an input image, the Discriminator outputs the likelihood of the image being real.
# Binary classification - true or false (1 or 0). So using sigmoid activation.

# Unlike regular GAN here we are also providing number of classes as input.
# Input to the model will be both images and labels.
def define_discriminator(in_shape=(128, 9, 1), n_classes=6):
    # label input
    in_label = Input(
        shape=(1,), name='Discriminator-Label-Input-Layer')  # Shape 1
    # embedding for categorical input
    # each label (total 10 classes for cifar), will be represented by a vector of size 50.
    # This vector of size 50 will be learnt by the discriminator
    li = Embedding(
        n_classes, 50, name='Discriminator-Label-Embedding-Layer')(in_label)  # Shape 1,50
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]  # 128x9 = 1152.
    # Shape = 1, 3424
    li = Dense(n_nodes, name='Discriminator-Label-Dense-Layer')(li)
    # li = BatchNormalization(momentum=0.5)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1], 1),
                 name='Discriminator-Label-Reshape-Layer')(li)  # 128x9x1

    # image input
    in_image = Input(
        shape=in_shape, name='Discriminator-Image-Input-Layer')  # 128x9x1
    # concat label as a channel
    # 128x9x2 (2 channels, 1 for image and the 1 for labels)
    merge = Concatenate(name='Discriminator-Combine-Layer')([in_image, li])

    # down-sample: This part is same as unconditional GAN upto the output layer.
    # We will combine input label with input image and supply as inputs to the model.
    fe = Conv2D(filters=64, kernel_size=(4,4), strides=(1, 1), padding='same', name='Discriminator-Hidden-Layer-1')(merge)  # 64x5x64
    fe = Dropout(0.8)(fe)
    # fe = BatchNormalization(momentum=0.5)(fe)
    fe = LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1')(fe)
    # down-sample
    fe = Conv2D(filters=64, kernel_size=(4,4), strides=(1, 1), padding='same', name='Discriminator-Hidden-Layer-2')(fe)  # 32x3x128
    # fe = BatchNormalization(momentum=0.5)(fe)
    fe = LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2')(fe)
    fe = Dropout(0.8)(fe)
    
    # flatten feature maps
    # 4992  (3*13*128=8192)
    fe = Flatten(name='Discriminator-Flatten-Layer')(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(16, name='Discriminator-Label-Hidden-Dense-Layer5')(fe)

    # output
    out_layer = Dense(1, activation='sigmoid',
                      name='Discriminator-Output-Layer')(fe)  # Shape=1

    # define model
    # Combine input label with input image and supply as inputs to the model.
    model = Model([in_image, in_label], out_layer, name='Discriminator')
    # compile model
    #opt = Adam(lr=0.0002, beta_1=0.5)
    opt = SGD(learning_rate=0.0002)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model


# test_discr = define_discriminator()
# print(test_discr.summary())


def define_generator(latent_dim, n_classes=6):
    # label input
    # Input of dimension 1
    in_label = Input(shape=(1,), name='Generator-Label-Input-Layer')
    # embedding for categorical input
    # each label (total 18 classes for Opportunity), will be represented by a vector of size 50.
    # Shape 1,50
    li = Embedding(
        n_classes, 50, name='Generator-Label-Embedding-Layer')(in_label)

    # linear multiplication
    # To match the dimensions for concatenation later in this step.
    n_nodes = 16 * 9
    li = Dense(n_nodes, name='Generator-Label-Dense-Layer')(li)  # 1,64
    # reshape to additional channel
    li = Reshape((16, 9, 1), name='Generator-Label-Reshape-Layer')(li)

    # image generator input
    # Input of dimension 100
    in_lat = Input(shape=(latent_dim,), name='Generator-Latent-Input-Layer')

    # foundation for 8x8 image
    # We will reshape input latent vector into 8x9 image as a starting point.
    # So n_nodes for the Dense layer can be 128x9x1 so when we reshape the output
    # it would be 8x8x128 and that can be slowly upscaled to 32x32 image for output.
    # Note that this part is same as unconditional GAN until the output layer.
    # While defining model inputs we will combine input label and the latent input.

    n_nodes = 16 * 9 * 16  # This Part is very Important

    # shape=8192
    gen = Dense(n_nodes, name='Generator-Foundation-Layer')(in_lat)
    gen = LeakyReLU(
        alpha=0.2, name='Generator-Foundation-Layer-Activation-1')(gen)
    # Shape=8x107x32
    gen = Reshape((16, 9, 16), name='Generator-Foundation-Layer-Reshape-1')(gen)

    # merge image gen and label input
    # Shape=32x9x17 (Extra channel corresponds to the label)
    merge = Concatenate(name='Generator-Combine-Layer')([gen, li])

    # up-sample to 16x16 ====== HIDDEN layer 1
    gen = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(
        2, 1), padding='same', name='Generator-Hidden-Layer-1')(merge)  # 32x9x32
    # gen = BatchNormalization(momentum=0.5)(gen)
    gen = LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-1')(gen)
    gen = Dropout(0.3, name='Generator-1stCONV2D-Layer-Dropout')(gen)
    #gen = UpSampling2D(size=(2, 1), data_format=None, interpolation="nearest")(gen)

    # up-sample to 32x32 ========== HIDDEN layer 1
    gen = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(
        2, 1), padding='same', name='Generator-Hidden-Layer-2')(gen)  # 64x9x64
    # gen = BatchNormalization(momentum=0.5)(gen)
    gen = LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-2')(gen)
    gen = Dropout(0.3, name='Generator-2ndCONV2D-Layer-Dropout')(gen)
    #gen = UpSampling2D(size=(2, 1), data_format=None, interpolation="nearest")(gen)
    
    # up-sample to 32x32 ========== HIDDEN layer 1
    gen = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(
        2, 1), padding='same', name='Generator-Hidden-Layer-3')(gen)  # 64x9x64
    # gen = BatchNormalization(momentum=0.5)(gen)
    gen = LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-3')(gen)
    gen = Dropout(0.3, name='Generator-3rdCONV2D-Layer-Dropout')(gen)
    # gen = UpSampling2D(size=(2, 1), data_format=None, interpolation="nearest")(gen)
    
    # output
    out_layer = Conv2D(filters=1, kernel_size=(8, 8), activation='tanh',
                       padding='same', name='Generator-Output-Layer')(gen)  # 32x107x1
    # define model
    model = Model([in_lat, in_label], out_layer, name='Generator')
    # Model not compiled as it is not directly trained like the discriminator.
    return model


# test_gen = define_generator(100, n_classes=18)
# print(test_gen.summary())


# #Generator is trained via GAN combined model.
# define the combined generator and discriminator model, for updating the generator
# Discriminator is trained separately so here only generator will be trained by keeping
# the discriminator constant.
def define_gan(g_model, d_model):
    # Discriminator is trained separately. So set to not trainable.
    d_model.trainable = False

    # connect generator and discriminator...
    # first, get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input  # Latent vector size and label size
    # get image output from the generator model
    gen_output = g_model.output  # 32x32x3

    # generator image output and corresponding input label are inputs to discriminator
    gan_output = d_model([gen_output, gen_label])

    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)

    # compile model
    opt = Adam(learning_rate=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# # select real samples
# pick a batch of random real samples to train the GAN
# In fact, we will train the GAN on a half batch of real images and another
# half batch of fake images.
# For each real image we assign a label 1 and for fake we assign label 0.
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels and assign to y (don't confuse this with the above labels that correspond to cifar labels)
    y = ones((n_samples, 1))  # Label=1 indicating they are real
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=6):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    labels = labels.reshape(len(labels), 1)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
# Supply the generator, latent_dim and number of samples as input.
# Use the above latent point generator to generate latent points.
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))  # Label=0 indicating they are fake
    return [images, labels_input], y


def saveResultsCSV(gen_activation, epochs, batch_per_epoch, d_loss_real, d_loss_fake, g_loss):
    path = './cGAN/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = './cGAN/D_and_G_loss.csv'
    file = Path(fileString)
    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write(
            'gen_activation; epochs; batch_per_epoch; d_loss_real; d_loss_fake; g_loss\n')
        f.close()
    with open(fileString, "a") as f:
        f.write('{};{};{};{};{};{}\n'.format(gen_activation, epochs,
                batch_per_epoch, d_loss_real, d_loss_fake, g_loss))
    f.close()


# train the generator and discriminator
# We loop through a number of epochs to train our Discriminator by first selecting
# a random batch of images from our true/real dataset.
# Then, generating a set of images using the generator.
# Feed both set of images into the Discriminator.
# Finally, set the loss parameters for both the real and fake images, as well as the combined loss.

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # the discriminator model is updated for a half batch of real samples
    half_batch = int(n_batch / 2)
    # and a half batch of fake samples, combined a single batch.
    # manually enumerate epochs
    gen_activation = 'tanh'

    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # Train the discriminator on real and fake images, separately (half batch each)
            # Research showed that separate training is more effective.
            # get randomly selected 'real' samples
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(
                dataset, half_batch)

            # update discriminator model weights
            # train_on_batch allows you to update weights based on a collection
            # of samples you provide
            d_loss_real, _ = d_model.train_on_batch(
                [X_real, labels_real], y_real)

            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(
                g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)

            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss if you want to report single..

            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(
                latent_dim, n_batch)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            # This is where the generator is trying to trick discriminator into believing
            # the generated image is true (hence value of 1 for y)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # Generator is part of combined model where it got directly linked with the discriminator
            # Train the generator with latent_dim as x and 1 as y.
            # Again, 1 as the output as it is adversarial and if generator did a great
            # job of folling the discriminator then the output would be 1 (true)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # Print losses on this batch
            print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f' % (
                i + 1, j + 1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
            saveResultsCSV(gen_activation, i, j,
                           d_loss_real, d_loss_fake, g_loss)
            # if (i % 2500) == 0:
            #    g_model.save(f'CGAN_OPPO_{i}epochs.h5')

    # save the generator model
    g_model.save(f'cGAN_HAR_{n_epochs}_epochs_{n_batch}_Batch.h5')


# Train the GAN

# size of the latent space, number of Batch and number of Epochs

# ********************************************************************
latent_dim = 100  # *
n_batch = 64  # *
n_epochs = 200  # *
# ********************************************************************

# create the discriminator
d_model = define_discriminator()

# create the generator
g_model = define_generator(latent_dim)

# create the gan
gan_model = define_gan(g_model, d_model)
gan_model.summary()
# Train Model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch)
