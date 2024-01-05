import gc
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data_preprocessing import *
from models import *
from save_result import *
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# -------------------------------------------Importing the preprocessed data----------------
eeg_valence_slided, valence_slided, eeg_arousal_slided, arousal_slided, combined_data_valence_slided, combined_valence_slided, combined_data_arousal_slided, combined_arousal_slided = data_pre_pro_with_scalling()

print('Shape of eeg_valence_data:', eeg_valence_slided.shape)
print('Shape of valence_labels:', valence_slided.shape)
print('Shape o eeg_arousal_data:', eeg_arousal_slided.shape)
print('Shape of arousal_labels:', arousal_slided.shape)
print('Shape of combined_data_valence_slided:',combined_data_valence_slided.shape)
print('Shape of combined_valence_labels:', combined_valence_slided.shape)
print('Shape of combined_data_arousal_slided:',combined_data_arousal_slided.shape)
print('Shape of combined_arousal_labels:', combined_arousal_slided.shape)

label_name = input('Which label (valence or arousal) do you want to use for the computation?\n')
if label_name == "valence":
    x = combined_data_valence_slided
    y = combined_valence_slided
elif label_name == "arousal":
    x = combined_data_arousal_slided
    y = combined_arousal_slided

# Considering only Valence
method = 'None'
activation = 'relu'
init_mode = 'glorot_uniform'
optimizer = 'Adam'
dropout_rate = 0.6
batch_size = 32
epochs = 200
verbose = 2
modelName = 'CONV2D'
Aug_factor = "None"

# -----------------------------------------------------------------Using early stop and Model Check point------------------------------------------------------------------------
es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=100)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = 0
best_f1_score = 0
best_train_accuracy = 0
best_test_accuracy = 0
all_accuracies = []
all_f1_scores = []
n_splits = 3  # You can adjust this number

# Initialize K-fold cross-validator
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_index, test_index in tqdm(kf.split(x, y), total=n_splits, desc="K-Fold Cross-validation"):
    x_train, x_test = combined_data_arousal_slided[train_index], combined_data_arousal_slided[test_index]
    y_train, y_test = combined_arousal_slided[train_index], combined_arousal_slided[test_index]
    
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], 1
    model = conv2D(activation, init_mode, optimizer, dropout_rate, n_timesteps, n_features, n_outputs)
    print("Fit model:")

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=es)
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))
    predictions = (model.predict(x_test) > 0.5).astype("int32")
    f1score_macro = (f1_score(y_test, predictions, average="macro"))
    accuracy = accuracy_score(y_test, predictions)

    # Evaluate the model on the test set
    test_predictions = np.argmax(model.predict(x_test), axis=1)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions, average='macro')

    all_accuracies.append(accuracy)
    all_f1_scores.append(f1score_macro)
                
    # Update best train and test accuracy                
    if train_acc > best_train_accuracy:
        best_train_accuracy = train_acc
    if test_acc > best_test_accuracy:
        best_test_accuracy = test_acc

    # Update best accuracy and F1 score
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    if f1score_macro > best_f1_score:
        best_f1_score = f1score_macro
#Calculate the average accuracy and f1 score. 
average_accuracy = np.mean(all_accuracies)
average_f1_score = np.mean(all_f1_scores)
# Calculate standard deviation of average accuracy and average F1 score
std_accuracy = np.std(all_accuracies)
std_f1_score = np.std(all_f1_scores)

saveResultsCSV(label = label_name, aug_method = method, aug_factor=0, modelname = modelName, epochs = epochs, batch_size = batch_size, 
                           train_acc = best_train_accuracy, test_acc = best_test_accuracy, best_f1score_macro = best_f1_score, 
                           avg_f1score = average_f1_score, std_f1score = std_f1_score,  best_accuracy=best_accuracy, avg_acc = average_accuracy, 
                           std_acc = std_accuracy ) # In save results we are providing the AUG factor

gc.collect()
