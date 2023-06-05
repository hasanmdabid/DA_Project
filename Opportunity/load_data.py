
import os.path
from pathlib import Path
from datetime import datetime
import pickle

def load_data(x):
    import pandas as pd
    x = pd.read_csv(f'~/Documents/DA_Project/OPPORTUNITY/Data/{x}.csv',dtype='float32')
    x = x.fillna(method='ffill')  # Replacing the 'Nan' values with 0 in the dataset
    # x = x[(~x.astype('bool')).mean(axis=1) < 0.10]  # Dropping any rows that contains 90% of its overall column values is equal to 0
    return x


def saveResultsCSV(methode, modelName, epochs, batch_size, accuracy, fscore, scoresFMeasure, modelLayers, time_elapsed,
                   nKerns, filterSizes):
    path = './results/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = './results/Without_Majority_class_results.csv'
    file = Path(fileString)

    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write(
            'Finished on; Methode; Modelname; Epochs; Batch_Size; accuracy; fscore; fscores; layers; Time elapsed ('
            'hh:mm:ss.ms); nKerns; filterSize\n')
        f.close()
    with open(fileString, "a") as f:
        f.write('{};{};{};{};{};{};{};{};{};{};{};{}\n'.format(now, methode, modelName, epochs, batch_size, accuracy,
                                                               fscore, scoresFMeasure, modelLayers, time_elapsed,
                                                               nKerns, filterSizes))
    f.close()


def predict_on_model(model, testing_data, testing_labels):
    testing_pred = model.predict(testing_data)

    testing_pred = testing_pred.argmax(axis=-1)
    true_labels = testing_labels.argmax(axis=-1)

    return testing_pred, true_labels


def save_variable(path, filename, obj):
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + filename + '.pckl', 'wb')
    pickle.dump(obj, f)
    f.close()
    return True



