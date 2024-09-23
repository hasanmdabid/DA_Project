# pylint: disable-all
# This script will pergorm conditional GAN. To successfully run the script please follow the instruction below:
# 1st : run the "data_preprocessing.py file. It will preprocess the raw DEAP data and store the preprocessed data as "np" file at the "Deap" directory
# 2nd : If you run the script It will ask you, for which label do you want to train your GAN model. Either give the input label "arousal" or "valence"
# 3rd : The script will train the conditional GAN model Model. And record the GAN loss in the .Deap/cDEAPGAN/results/loss_record" folder.
# 4th : After the training the script will save the trained model to the  ".Deap/cDEAPGAN/results/trained_model" folder.
# 5th : To generate the synthetic data you will need to run the synthetic_data_Generator_DEAP.py file. This script will use the trained GAN model to genrate the synthetic data
# and save it to the "./Deap/Data/cGAN_generated_data/".
import gc
import numpy as np
import platform
from numpy.random import randn
from numpy.random import randint
from numpy import zeros
from numpy import ones
import tensorflow as tf
from data_preprocessing import *
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose, MaxPool2D, AveragePooling2D, BatchNormalization, UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils import plot_model # for plotting model diagram

import os.path
from pathlib import Path
from datetime import datetime


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

def check_gpu():

    if 'linux' in platform.platform().lower():
        print("Check GPU...")
        if len(tf.config.list_physical_devices('GPU')) == 0:
            print("GPU is not available!")
            quit()

        print("GPU is available!")

check_gpu()

# -------------------------------------------Importing the preprocessed numpy datadata----------------
# For this study we are importing ath combined (EEG and video) data and labels. 
import numpy as np
# Specify the folder and filenames
folder_path = '/home/abidhasan/Documents/DA_Project/Deap/Data/pre_processed_np_data/combined'
label_name = input("Enter your value: ")
if label_name == 'arousal':
    data_file_name = 'arousal_data.npy'
    label_file_name = 'arousal_label.npy'
elif label_name == 'valence':
    data_file_name = 'valence_data.npy'
    label_file_name = 'valence_label.npy'
# Combine folder path and file names
data_full_path = f'{folder_path}/{data_file_name}'
label_full_path = f'{folder_path}/{label_file_name}'

# Load data and label from separate files
data = np.load(data_full_path)
label = np.load(label_full_path)

print('Data sahpe:')
print(data.shape)
print('\nLavels shape:')
print(label.shape)

max_val_sacled = np.max(data)
min_val_scaled = np.min(data)
print('Maximume_value Before rescaled:', max_val_sacled)
print('Minimume_value Before rescaled:', min_val_scaled)


save_pic_dir = '/home/abidhasan/Documents/DA_Project/Deap/cDEAPGAN/figures'
save_model_dir = '/home/abidhasan/Documents/DA_Project/Deap/cDEAPGAN/results/trained_model'
save_loss_dir = '/home/abidhasan/Documents/DA_Project/Deap/cDEAPGAN/results/loss_record'

########################################################################################################################
# Preparing the dataset to use for the CGAN
# Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
# Here I am Using The MinMaxScaler to normalize the data
# 1st I have reshaped the 3D trainx to 2D then perform the MinMaxsacler
# then I reshape it to 4D (Nr_samples, Nr_rows, Nr_columns, Nr_channels)
# Here channels = 1 (For Image channels = 3 for Colored image and 1 for the Gray scale Image)
# The dimension the of the Trainx is 4d (Sample, Row, column, channels) and labels is Column Vector (Samples, 1)

from sklearn.preprocessing import MinMaxScaler
nr_samples, nr_rows, nr_columns = data.shape
data = data.reshape(nr_samples * nr_rows, nr_columns)  # To sclae the data converting from 2D o 3D
scaler = MinMaxScaler(feature_range=(-1, 1))  # Declaring the scaler
scaler.fit(data)  # Fitting the scaler
data_scaled = scaler.transform(data)  # Transforming the DATA

from joblib import dump
dump(scaler,
     f'/home/abidhasan/Documents/DA_Project/Deap/cDEAPGAN/{label_name}_minmax_scaler.bin', compress=True)


# --------- Checking whether all the Feature value is scaled between +1 and -1 ------------------------------------------
max_val = np.max(data_scaled)
min_val = np.min(data_scaled)
print('Maximume_value after scaled:', max_val)
print('Minimume_value after scaled:', min_val)

# Convert the data to 4D array (nr. SAMPLES, nr. ROWS, nr. COLUMNS, nr. CHANNELS)

data = data_scaled.reshape(nr_samples, nr_rows, nr_columns, 1)
print("Shape of the scaled array: ", data.shape)
label = np.expand_dims(label, axis = 1)
dataset = [data, label]
print(dataset[0].shape)
print(dataset[1].shape)
values, counts = np.unique(label, return_counts=True)
print(values, counts)


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

def define_discriminator(in_shape=(128, 40, 1), n_classes=2):
    # label input
    in_label = Input(shape=(1,), name='Discriminator-Label-Input-Layer')  # Shape 1
    # embedding for categorical input
    # each label (total 10 classes for cifar), will be represented by a vector of size 50.
    # This vector of size 50 will be learnt by the discriminator
    li = Embedding(n_classes, 50, name='Discriminator-Label-Embedding-Layer')(in_label)  # Shape 1,50
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]  # 128x40 = 5120.
    li = Dense(n_nodes, name='Discriminator-Label-Dense-Layer')(li)  # Shape = 1, 5120
    #li = BatchNormalization(momentum=0.4)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1], 1), name='Discriminator-Label-Reshape-Layer')(li)  # 128x40x1

    # image input
    in_image = Input(shape=in_shape, name='Discriminator-Image-Input-Layer')  # 128x40x1
    # concat label as a channel
    merge = Concatenate(name='Discriminator-Combine-Layer')([in_image, li])  # 128x40x2 (2 channels, 1 for image and the 1 for labels)

    # down-sample: This part is same as unconditional GAN upto the output layer.
    # We will combine input label with input image and supply as inputs to the model.
    fe = Conv2D(filters=64, kernel_size=(4, 1), strides=(2, 1), padding='same', name='Discriminator-Hidden-Layer-1')(merge)  # 16x54x64
    #fe = BatchNormalization(momentum=0.4)(fe)
    fe = LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1')(fe)
    fe = AveragePooling2D(pool_size=(2, 1), strides=(2, 1), padding='same', name='Discriminator-MaxPool-Layer_1')(fe)  # 8
    # down-sample
    fe = Conv2D(filters=128, kernel_size=(4, 1), strides=(2, 1), padding='same', name='Discriminator-Hidden-Layer-2')(fe)  # 8x27x128
    #fe = BatchNormalization(momentum=0.4)(fe)
    fe = LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2')(fe)
    fe = AveragePooling2D(pool_size=(2, 1), strides=(2, 1), padding='same', name='Discriminator-MaxPool-Layer-2')(fe)  # Max Pool 3*13*128

    # flatten feature maps
    fe = Flatten(name='Discriminator-Flatten-Layer')(fe)  # 4992  (3*13*128=8192)


    # Dense Layer
    # fe = Dense(256, name='Discriminator-Label-Hidden-Dense-Layer2')(fe)
    fe = Dense(128, name='Discriminator-Label-Hidden-Dense-Layer1')(fe)
    # fe = Dense(256, name='Discriminator-Label-Hidden-Dense-Layer2')(fe)
    fe = Dense(64, name='Discriminator-Label-Hidden-Dense-Layer5')(fe)
        # output
    out_layer = Dense(1, activation='sigmoid', name='Discriminator-Output-Layer')(fe)  # Shape=1

    # define model
    ##Combine input label with input image and supply as inputs to the model.
    model = Model([in_image, in_label], out_layer, name='Discriminator')
    # compile model
    opt = SGD(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Instantiate
dis_model = define_discriminator()

# Show model summary and plot model diagram
plot_model(dis_model, show_shapes=True, show_layer_names=True, dpi=400,
           to_file='/home/abidhasan/Documents/DA_Project/Deap/cDEAPGAN/figures/discriminator_structure.png')



def define_generator(latent_dim, n_classes=2):
    # label input
    in_label = Input(shape=(1,), name='Generator-Label-Input-Layer')  # Input of dimension 1
    # embedding for categorical input
    # each label (total 2 classes for DEAP), will be represented by a vector of size 50.
    li = Embedding(n_classes, 50, name='Generator-Label-Embedding-Layer')(in_label)  # Shape 1,50

    # linear multiplication
    n_nodes = 8 * 40  # To match the dimensions for concatenation later in this step.
    li = Dense(n_nodes, name='Generator-Label-Dense-Layer')(li)  # 1,64
    # reshape to additional channel
    li = Reshape((8, 40, 1), name='Generator-Label-Reshape-Layer')(li)


    # image generator input
    in_lat = Input(shape=(latent_dim,), name='Generator-Latent-Input-Layer')  # Input of dimension 100

    # foundation for 8x8 image
    # We will reshape input latent vector into 8x8 image as a starting point.
    # So n_nodes for the Dense layer can be 128x8x8 so when we reshape the output
    # it would be 8x8x128 and that can be slowly upscaled to 32x32 image for output.
    # Note that this part is same as unconditional GAN until the output layer.
    # While defining model inputs we will combine input label and the latent input.

    n_nodes = 8 * 40 * 2  # This Part is very Important

    gen = Dense(n_nodes, name='Generator-Foundation-Layer')(in_lat)  # shape=81920
    gen = LeakyReLU(alpha=0.2, name='Generator-Foundation-Layer-Activation-1')(gen)
    gen = Reshape((8, 40, 2), name='Generator-Foundation-Layer-Reshape-1')(gen)  # Shape=8x107x32

    # merge image gen and label input
    merge = Concatenate(name='Generator-Combine-Layer')([gen, li])  # Shape=8x40x654 (Extra channel corresponds to the label)

    # up-sample to 16x16 ====== HIDDEN layer 1
    gen = Conv2DTranspose(filters=64, kernel_size=(4, 1), strides=(2, 1), padding='same', name='Generator-Hidden-Layer-1')(merge)  # 16x107x64
    #gen = BatchNormalization(momentum=0.4)(gen)
    gen = LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-1')(gen)
    gen = Dropout(0.4, name='Generator-1stCONV2D-Layer-Dropout')(gen)
    
    # up-sample to 16x16 ====== HIDDEN layer 1
    gen = Conv2DTranspose(filters=64, kernel_size=(4, 1), strides=(2, 1), padding='same', name='Generator-Hidden-Layer-2')(gen)  # 16x107x64
    #gen = BatchNormalization(momentum=0.4)(gen)
    gen = LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-2')(gen)
    gen = Dropout(0.4, name='Generator-2ndCONV2D-Layer-Dropout')(gen)
    
        # up-sample to 16x16 ====== HIDDEN layer 1
    gen = Conv2DTranspose(filters=64, kernel_size=(4, 1), strides=(2, 1), padding='same', name='Generator-Hidden-Layer-3')(gen)  # 16x107x64
    #gen = BatchNormalization(momentum=0.4)(gen)
    gen = LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-3')(gen)
    gen = Dropout(0.4, name='Generator-3rdCONV2D-Layer-Dropout')(gen)

    # up-sample to 32x32 ========== HIDDEN layer 1
    gen = Conv2DTranspose(filters=32, kernel_size=(4, 1), strides=(2, 1), padding='same', name='Generator-Hidden-Layer-4')(gen)  # 32x107x64
    #gen = BatchNormalization(momentum=0.4)(gen)
    gen = LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-4')(gen)
    gen = Dropout(0.4, name='Generator-4thCONV2D-Layer-Dropout')(gen)

    # output
    out_layer = Conv2D(filters=1, kernel_size=(3, 1), strides =(1,1), activation='tanh', padding='same', name='Generator-Output-Layer')(gen)  # 32x107x1
    # define model
    model = Model([in_lat, in_label], out_layer, name='Generator')
    return model  # Model not compiled as it is not directly trained like the discriminator.

gen_model = define_generator(latent_dim =100, n_classes=2)
# Show model summary and plot model diagram
plot_model(gen_model, show_shapes=True, show_layer_names=True, dpi=400,
           to_file='/home/abidhasan/Documents/DA_Project/Deap/cDEAPGAN/figures/generator_structure.png')


def define_gan(g_model, d_model):
    d_model.trainable = False  # Discriminator is trained separately. So set to not trainable.

    ## connect generator and discriminator...
    # first, get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input  # Latent vector size and label size
    # get image output from the generator model
    gen_output = g_model.output  # 32x32x3

    # generator image output and corresponding input label are inputs to discriminator
    gan_output = d_model([gen_output, gen_label])

    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)

    # compile model
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
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
def generate_latent_points(latent_dim, n_samples, n_classes=2):
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


def saveResultsCSV(gen_activation, epochs, batch_per_epoch, d_loss_real, d_loss_fake, g_loss, label_name):
    #path = './results/loss_record/'
    path = save_loss_dir+'/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = save_loss_dir + f'/{label_name}_D_and_G_loss.csv'
    file = Path(fileString)
    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write('gen_activation; epochs; batch_per_epoch; d_loss_real; d_loss_fake; g_loss\n')
        f.close()
    with open(fileString, "a") as f:
        f.write('{};{};{};{};{};{}\n'.format(gen_activation, epochs, batch_per_epoch, d_loss_real, d_loss_fake, g_loss))
    f.close()
    
    
    
        
# train the generator and discriminator
# We loop through a number of epochs to train our Discriminator by first selecting
# a random batch of images from our true/real dataset.
# Then, generating a set of images using the generator.
# Feed both set of images into the Discriminator.
# Finally, set the loss parameters for both the real and fake images, as well as the combined loss.

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch, label_name):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)  # the discriminator model is updated for a half batch of real samples
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
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)

            # update discriminator model weights
            ##train_on_batch allows you to update weights based on a collection
            # of samples you provide
            d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)

            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)

            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss if you want to report single..

            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)

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
            saveResultsCSV(gen_activation, i, j, d_loss_real,
                           d_loss_fake, g_loss, label_name)
            # if (i % 2500) == 0:
            #    g_model.save(f'CGAN_DEAP_{i}epochs.h5')

    # save the generator model
    g_model.save(save_model_dir + f'/CGAN_DEAP_arousal_{n_epochs}epochs_{n_batch}Batch.h5')
    
# Train the GAN

# size of the latent space, number of Batch and number of Epochs

# ********************************************************************
latent_dim = 100                                                    #*
n_batch = 64                                                        #*
n_epochs = 500                                                      #*
# ********************************************************************

# create the discriminator
d_model = define_discriminator()

# create the generator
g_model = define_generator(latent_dim)

# create the gan
gan_model = define_gan(g_model, d_model)
# Train Model
train(g_model, d_model, gan_model, dataset,
      latent_dim, n_epochs, n_batch, label_name)
