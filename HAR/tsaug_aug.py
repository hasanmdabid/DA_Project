import numpy as np
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve, Pool, AddNoise

def tsaug(aug_method, factor):
    if aug_method == "convolve":
        # Convolve time series with a kernel window OF 16.
        aug = (Convolve(window="flattop", size=16) * factor)
    elif aug_method == "quantize":
        # random quantize to 10-, 20-, or 30- level sets
        aug = (Quantize(n_levels=[10, 20, 30]) * factor)
    elif aug_method == "drift" or aug_method == "scaling":
        # with 80% probability, random drift the signal up to 10% - 50%
        aug = (Drift(max_drift=(0.1, 0.5)) @ 0.8 * factor)
    elif aug_method == "pool":
        # Reduce the temporal resolution without changing the length
        aug = (Pool(size=10) * factor)
    elif aug_method == "TW":
        aug = (TimeWarp() * factor)
    return aug
