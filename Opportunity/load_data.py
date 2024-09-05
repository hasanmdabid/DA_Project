# pylint: disable-all
# To use theis repository for Opportunity dataset one must have to download the "opportunity" dataset from the following link. 
# https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition
# The dataset is public and anyone can use it without any prior consent. 
 # To read the dataset, please save the csv files under teh following folders "./opportunity/Data/{x}.csv"
 

import os.path
from pathlib import Path
from datetime import datetime
import pickle

def load_data(x):
    import pandas as pd
    x = pd.read_csv(f'~/Documents/DA_Project/Opportunity/Data/{x}.csv',dtype='float32')
    x = x.fillna(method='ffill')  # Replacing the 'Nan' values with 0 in the dataset
    return x


def predict_on_model(model, testing_data, testing_labels):
    testing_pred = model.predict(testing_data)

    testing_pred = testing_pred.argmax(axis=-1)
    true_labels = testing_labels.argmax(axis=-1)

    return testing_pred, true_labels



