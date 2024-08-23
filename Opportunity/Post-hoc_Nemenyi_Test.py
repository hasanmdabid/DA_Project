# pylint: disable=all

import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import numpy as np

# Step 1: Read the CSV file
df = pd.read_csv('Opportunity.csv', delimiter=';')

# Step 2: Extract the required columns
methods = df['Method'].unique()
results_acc = {method: [] for method in methods}
results_fscore = {method: [] for method in methods}

for method in methods:
    results_acc[method] = df[df['Method'] == method]['accuracy'].values
    results_fscore[method] = df[df['Method'] ==
                                method]['Average_fscore_Macro'].values

# Step 3: Perform the Friedman test
data_acc = [results_acc[method] for method in methods]
data_fscore = [results_fscore[method] for method in methods]

friedman_acc = friedmanchisquare(*data_acc)
friedman_fscore = friedmanchisquare(*data_fscore)

print('Friedman test for accuracy:', friedman_acc)
print('Friedman test for f-score:', friedman_fscore)

# Step 4: Conduct the post-hoc Nemenyi test
data_acc_combined = np.concatenate([results_acc[method] for method in methods])
groups_acc = np.concatenate(
    [[method] * len(results_acc[method]) for method in methods])

data_fscore_combined = np.concatenate(
    [results_fscore[method] for method in methods])
groups_fscore = np.concatenate(
    [[method] * len(results_fscore[method]) for method in methods])

nemenyi_acc = sp.posthoc_nemenyi_friedman(data_acc_combined, groups_acc)
nemenyi_fscore = sp.posthoc_nemenyi_friedman(
    data_fscore_combined, groups_fscore)

print('Nemenyi test for accuracy:')
print(nemenyi_acc)
print('Nemenyi test for f-score:')
print(nemenyi_fscore)
