# pylint: disable-all
from collections import Counter
import smote_variants as sv
import numpy as np
import pandas as pd
import statistics
import math

def SMOTE(data):

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    counter = Counter(y)

    max_value = max(counter.values())
    print(max_value)
    mean = math.floor(statistics.mean(list(counter.values())))

    strategy = {}
    for key in counter:
        print(key)
        if key == 0:
            del key
        else:
            strategy[key] = math.floor(max_value / 2)

    # print('Strategy:', strategy)
    # oversample = BorderlineSMOTE(random_state=10, kind='borderline-1') # <-------- Using Bordeline SMOTE
    # oversample = SMOTE(sampling_strategy=strategy)                     # <-------- Using the Regular SMOTE with Samply strategy
    # oversample = SMOTE()                                               # <-------- Using the Regular SMOTE without Samply strategy
    oversample = sv.MulticlassOversampling(oversampler='distance_SMOTE', oversampler_params={'random_state': 5})

    x_samp, y_samp = oversample.sample(x, y)

    # print('shape of x_SMOTE:', x_samp.shape)
    # print('shape of y_SMOTE:', y_samp.shape)

    # counter = Counter(y_samp)
    # print(counter)


    # x_samp = x_samp.to_numpy()
    # y_samp = y_samp.to_numpy()

    data = np.concatenate((x_samp, y_samp[:, None]), axis=1)
    df = pd.DataFrame(data)
    # print(df.head(5))

    return df



