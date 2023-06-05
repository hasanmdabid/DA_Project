import numpy as np
import os
from utils.input_data import read_data_sets
import utils.datasets as ds
import utils.augmentation as aug
import utils.helper as hlp
import matplotlib.pyplot as plt

dataset = "CBF"

nb_class = ds.nb_classes(dataset)
nb_dims = ds.nb_dims(dataset)

# Load Data
train_data_file = os.path.join("data", dataset, "%s_TRAIN.tsv"%dataset)
test_data_file = os.path.join("data", dataset, "%s_TEST.tsv"%dataset)

x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")

y_train = ds.class_offset(y_train, dataset)
y_test= ds.class_offset(y_test, dataset)
nb_timesteps = int(x_train.shape[1] / nb_dims)
input_shape = (nb_timesteps , nb_dims)

x_train_max = np.max(x_train)
x_train_min = np.min(x_train)
x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
# Test is secret
x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.

x_test = x_test.reshape((-1, input_shape[0], input_shape[1])) 
x_train = x_train.reshape((-1, input_shape[0], input_shape[1])) 
print("Shape of the X train data:", x_train.shape)
print("Shape of the X Test data:", x_test.shape)
print("Shape of the Y train:", y_train.shape)
print(y_train)
print("Shape of the Y test:", y_test.shape)

jitter_x = aug.jitter(x_train)
print("Shape of Augmented by Jitter:", jitter_x.shape)
random_DTW = aug.random_guided_warp(x_train, y_train)
print("Shape of Augmented by Random DTW:", random_DTW.shape)
guided_DTW_1 = aug.discriminative_guided_warp(x_train, y_train)
guided_DTW_2 = aug.discriminative_guided_warp(x_train, y_train)
print("Shape of Augmented by Guided DTW:", guided_DTW_1.shape)
comparison = guided_DTW_1 == guided_DTW_2
equal_arrays = comparison.all()
print(equal_arrays)
dist = np.linalg.norm(guided_DTW_1 - guided_DTW_2)
 
# printing Euclidean distance
print(dist)