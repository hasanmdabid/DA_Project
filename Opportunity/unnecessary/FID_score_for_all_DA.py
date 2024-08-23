import pandas as pd
import numpy as np
import load_data
from matplotlib import pyplot as plt
import seg_SMOTE
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import seg_TSAUG
import synthetic_data_generator
from DTW import TW, DGW, RGW
import utils.augmentation as aug


s1r1 = load_data.load_data("S1-ADL1")
s1r2 = load_data.load_data("S1-ADL2")
s1r3 = load_data.load_data("S1-ADL3")
s1r4 = load_data.load_data("S1-ADL4")
s1r5 = load_data.load_data("S1-ADL5")
s1_drill = load_data.load_data("S1-Drill")
s2r1 = load_data.load_data("S2-ADL1")
s2r2 = load_data.load_data("S2-ADL2")
s2r3 = load_data.load_data("S2-ADL3")
s2r4 = load_data.load_data("S2-ADL4")
s2r5 = load_data.load_data("S2-ADL5")
s2_drill = load_data.load_data("S2-Drill")
s3r1 = load_data.load_data("S3-ADL1")
s3r2 = load_data.load_data("S3-ADL2")
s3r3 = load_data.load_data("S3-ADL3")
s3r4 = load_data.load_data("S3-ADL4")
s3r5 = load_data.load_data("S3-ADL5")
s3_drill = load_data.load_data("S1-Drill")
s4r1 = load_data.load_data("S4-ADL1")
s4r2 = load_data.load_data("S4-ADL2")
s4r3 = load_data.load_data("S4-ADL3")
s4r4 = load_data.load_data("S4-ADL4")
s4r5 = load_data.load_data("S4-ADL5")
s4_drill = load_data.load_data("S4-Drill")

def column_notation(data):
    data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                        '18',
                        '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                        '35',
                        '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51',
                        '52',
                        '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68',
                        '69',
                        '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85',
                        '86',
                        '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101',
                        '102',
                        '103', '104', '105', '106', '107', 'Activity_Label']
    data['Activity_Label'] = data['Activity_Label'].replace(
            [406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511, 406508,
             404508,
             408512, 407521, 405506], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    return data

s1r1 = column_notation(s1r1)
s1r2 = column_notation(s1r2)
s1r3 = column_notation(s1r3)
s1r4 = column_notation(s1r4)
s1r5 = column_notation(s1r5)
s1_drill = column_notation(s1_drill)
s2r1 = column_notation(s2r1)
s2r2 = column_notation(s2r2)
s2r3 = column_notation(s2r3)
s2r4 = column_notation(s2r4)
s2r5 = column_notation(s2r5)
s2_drill = column_notation(s2_drill)
s3r1 = column_notation(s3r1)
s3r2 = column_notation(s3r2)
s3r3 = column_notation(s3r3)
s3r4 = column_notation(s3r4)
s3r5 = column_notation(s3r5)
s3_drill = column_notation(s3_drill)
s4r1 = column_notation(s4r1)
s4r2 = column_notation(s4r2)
s4r3 = column_notation(s4r3)
s4r4 = column_notation(s4r4)
s4r5 = column_notation(s4r5)
s4_drill = column_notation(s4_drill)


##########################################################################

# Activities list
activities = {'Old label': [0, 406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511,
                      406508, 404508, 408512, 407521, 405506],
        'New label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        'Activity': ['No activity', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'close Door 2', 'Open fridge', 'Close fridge',
                     'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2',
                     'Close Drawer 2', 'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup', 'Toggle Switch']}
activities = pd.DataFrame(activities)
print(activities)
ACTIVITY = ['No Activity', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'close Door 2', 'Open fridge', 'Close fridge',
                'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2', 'Close Drawer 2',
                'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup', 'Toggle Switch']

data_train = np.concatenate((s1r1, s1r2, s1r3, s1_drill, s1r4, s1r5, s2r1, s2r2, s2_drill, s3r1, s3r2, s3_drill), axis=0)
data_val = np.concatenate((s2r3, s3r3), axis=0)
data_test = np.concatenate((s2r4, s2r5, s3r4, s3r5), axis=0)

#data =np.concatenate((data_train, data_test, data_val), axis=0) 
data =data_train
print('shape of of dataset:', data.shape)

# Converting the Numpy data to Pandas Dataframe 
df = pd.DataFrame(data)
df = column_notation(df) # Addiitng the Notation of column values. 


# Grouping by the PD dataframe

gk = df.groupby('Activity_Label')
df_No_activity = gk.get_group(0)
df_open_door_1 = gk.get_group(1)
df_open_door_2 = gk.get_group(2)
df_close_door_1 = gk.get_group(3)
df_close_door_2 = gk.get_group(4)
df_open_fridge = gk.get_group(5)
df_close_fridge  = gk.get_group(6)
df_open_dishwasher = gk.get_group(7)
df_close_dishwasher = gk.get_group(8)
df_open_dawer_1 = gk.get_group(9)
df_close_dawer_1 = gk.get_group(10)
df_open_dawer_2 = gk.get_group(11)
df_close_dawer_2 = gk.get_group(12)
df_open_dawer_3 = gk.get_group(13)
df_close_dawer_3 = gk.get_group(14)
df_clean_table = gk.get_group(15)
df_drink_from_cup = gk.get_group(16)
df_toggle_switch = gk.get_group(17)

# Seperating the data and lavels

No_activity, No_activity_label = df_No_activity.iloc[:, :-1], df_No_activity.iloc[:, [-1]]
open_door_1, open_door_1_label = df_open_door_1.iloc[:, :-1], df_open_door_1.iloc[:, [-1]]
open_door_2, open_door_2_label = df_open_door_2.iloc[:, :-1], df_open_door_2.iloc[:, [-1]]
close_door_1, close_door_1_label = df_close_door_1.iloc[:, :-1], df_close_door_1.iloc[:, [-1]]
close_door_2, close_door_2_label = df_close_door_2.iloc[:, :-1], df_close_door_2.iloc[:, [-1]]
open_fridge, open_fridge_label = df_open_fridge.iloc[:, :-1], df_open_fridge.iloc[:, [-1]]
close_fridge, close_fridge_label = df_close_fridge.iloc[:, :-1], df_close_fridge.iloc[:, [-1]]
open_dishwasher, open_dishwasher_label = df_open_dishwasher.iloc[:, :-1], df_open_dishwasher.iloc[:, [-1]]
close_dishwasher, close_dishwasher_label = df_close_dishwasher.iloc[:, :-1], df_close_dishwasher.iloc[:, [-1]]
open_dawer_1, open_dawer_1_label = df_open_dawer_1.iloc[:, :-1], df_open_dawer_1.iloc[:, [-1]]
close_dawer_1, close_dawer_1_label = df_close_dawer_1.iloc[:, :-1], df_close_dawer_1.iloc[:, [-1]]
open_dawer_2, open_dawer_2_label = df_open_dawer_2.iloc[:, :-1], df_open_dawer_2.iloc[:, [-1]]
close_dawer_2, close_dawer_2_label = df_close_dawer_2.iloc[:, :-1], df_close_dawer_2.iloc[:, [-1]]
open_dawer_3, open_dawer_3_label = df_open_dawer_3.iloc[:, :-1], df_open_dawer_3.iloc[:, [-1]]
close_dawer_3, close_dawer_3_label = df_close_dawer_3.iloc[:, :-1], df_close_dawer_3.iloc[:, [-1]]
clean_table, clean_table_label = df_clean_table.iloc[:, :-1], df_clean_table.iloc[:, [-1]]
drink_from_cup, df_drink_from_cup_label = df_drink_from_cup.iloc[:, :-1], df_drink_from_cup.iloc[:, [-1]]
toggle_switch, toggle_switch_label = df_toggle_switch.iloc[:, :-1], df_toggle_switch.iloc[:, [-1]]

# Converting the Data into numpy array to find the FID score
No_activity = No_activity.to_numpy()
open_door_1 = open_door_1.to_numpy()
open_door_2 = open_door_2.to_numpy()
close_door_1 = close_door_1.to_numpy()
close_door_2 = close_door_2.to_numpy()
open_fridge = open_fridge.to_numpy()
close_fridge = close_fridge.to_numpy()
open_dishwasher = open_dishwasher.to_numpy()
close_dishwasher = close_dishwasher.to_numpy()
open_dawer_1 = open_dawer_1.to_numpy()
close_dawer_1 = close_dawer_1.to_numpy()
open_dawer_2 = open_dawer_2.to_numpy()
close_dawer_2 = close_dawer_2.to_numpy()
open_dawer_3 = open_dawer_3.to_numpy
close_dawer_3 = close_dawer_3.to_numpy()
clean_table = clean_table.to_numpy()
drink_from_cup = drink_from_cup.to_numpy()
toggle_switch = toggle_switch.to_numpy()


# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean) * 0.01
	return fid


def aug_method(method):
    #---------------------------------------------------->Generating data from method<-------------------------------------------------------------------------
    if method == 'SMOTE':
        s1r1_SMOTE = seg_SMOTE.seg_SMOTE(s1r1)
        s1r2_SMOTE = seg_SMOTE.seg_SMOTE(s1r2)
        s1r3_SMOTE = seg_SMOTE.seg_SMOTE(s1r3)
        s1r4_SMOTE = seg_SMOTE.seg_SMOTE(s1r4)
        s1r5_SMOTE = seg_SMOTE.seg_SMOTE(s1r5)
        s1_drill_SMOTE = seg_SMOTE.seg_SMOTE(s1_drill)
        s2r1_SMOTE = seg_SMOTE.seg_SMOTE(s2r1)
        s2r2_SMOTE = seg_SMOTE.seg_SMOTE(s2r2)
        s2_drill_SMOTE = seg_SMOTE.seg_SMOTE(s2_drill)
        s3r1_SMOTE = seg_SMOTE.seg_SMOTE(s3r1)
        s3r2_SMOTE = seg_SMOTE.seg_SMOTE(s3r2)
        s3_drill_SMOTE = seg_SMOTE.seg_SMOTE(s3_drill)        
        data_train_SMOTE = np.concatenate((s1r1_SMOTE, s1r2_SMOTE, s1r3_SMOTE, s1_drill_SMOTE, s1r4_SMOTE, s1r5_SMOTE, s2r1_SMOTE, s2r2_SMOTE, s2_drill_SMOTE, s3r1_SMOTE, s3r2_SMOTE, s3_drill_SMOTE), axis=0)      
        data_aug =data_train_SMOTE
        
        print('shape of of dataset:', data_aug.shape)

    elif method == 'TSAUG':
        trainxs1r1, trainys1r1 = seg_TSAUG.seg_TSAUG(s1r1)
        trainxs1r2, trainys1r2 = seg_TSAUG.seg_TSAUG(s1r2)
        trainxs1r3, trainys1r3 = seg_TSAUG.seg_TSAUG(s1r3)
        trainxs1r4, trainys1r4 = seg_TSAUG.seg_TSAUG(s1r4)
        trainxs1r5, trainys1r5 = seg_TSAUG.seg_TSAUG(s1r5)
        trainxs1_drill, trainys1_drill = seg_TSAUG.seg_TSAUG(s1_drill)
        trainxs2r1, trainys2r1 = seg_TSAUG.seg_TSAUG(s2r1)
        trainxs2r2, trainys2r2 = seg_TSAUG.seg_TSAUG(s2r2)
        trainxs2_drill, trainys2_drill = seg_TSAUG.seg_TSAUG(s2_drill)
        trainxs3r1, trainys3r1 = seg_TSAUG.seg_TSAUG(s3r1)
        trainxs3r2, trainys3r2 = seg_TSAUG.seg_TSAUG(s3r2)
        trainxs3_drill, trainys3_drill = seg_TSAUG.seg_TSAUG(s3_drill)
        data_TSAUG_3D = np.concatenate(
                (trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
                 trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)
        label_TSAUG = np.concatenate(
                (trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
                 trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)
        print('shape of 3D DATA_TSAUG =', data_TSAUG_3D.shape)
        print('shape of 3D LABELS_TSAUG =', label_TSAUG.shape)
        data_TSAUG = data_TSAUG_3D.reshape(data_TSAUG_3D.shape[0] * data_TSAUG_3D.shape[1], data_TSAUG_3D.shape[2])
        #eeg_data = eeg_data.reshape(eeg_data.shape[0] * eeg_data.shape[1], eeg_data.shape[2])
        label_TSAUG = np.repeat(label_TSAUG, 32, axis=0)  # Converting the labels into total number of time stamp
        label_TSAUG = label_TSAUG.reshape(label_TSAUG.shape[0], 1)
        print('shape of 2D DATA_TSAUG =', data_TSAUG.shape)
        print('shape of 3D LABELS_TSAUG =', label_TSAUG.shape)
        
        data_aug = np.concatenate((data_TSAUG, label_TSAUG), axis=1)      
        print('Shape of data:', data_aug.shape)
    
    elif method == 'GAN':
        data_GAN_3D, label_GAN = synthetic_data_generator.GAN_generator()
        print('Shape of GAN Generated X and Y', data_GAN_3D.shape, label_GAN.shape)
        data_GAN = data_GAN_3D.reshape(data_GAN_3D.shape[0] * data_GAN_3D.shape[1], data_GAN_3D.shape[2])
        label_GAN = np.repeat(label_GAN, 32, axis=0)  # Converting the labels into total number of time stamp
        label_GAN = label_GAN.reshape(label_GAN.shape[0], 1)
        print('shape of 2D DATA_GAN =', data_GAN.shape)
        print('shape of 2D LABELS_GAN =', label_GAN.shape)
        data_GAN = np.concatenate((data_GAN, label_GAN), axis=1) 
        data_aug = np.concatenate((data, data_GAN), axis=0)   
        print('Shape of GAN data:', data_aug.shape)
        
    elif method == 'random_guided_warp':
        trainxs1r1, trainys1r1 = RGW(s1r1)
        trainxs1r2, trainys1r2 = RGW(s1r2)
        trainxs1r3, trainys1r3 = RGW(s1r3)
        trainxs1_drill, trainys1_drill = RGW(s1_drill)
        trainxs1r4, trainys1r4 = RGW(s1r4)
        trainxs1r5, trainys1r5 = RGW(s1r5)
        trainxs2r1, trainys2r1 = RGW(s2r1)
        trainxs2r2, trainys2r2 = RGW(s2r2)
        trainxs2_drill, trainys2_drill = RGW(s2_drill)
        trainxs3r1, trainys3r1 = RGW(s3r1)
        trainxs3r2, trainys3r2 = RGW(s3r2)
        trainxs3_drill, trainys3_drill = RGW(s3_drill)
        trainx = np.concatenate((trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
                 trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)
        trainy = np.concatenate((trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
                 trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)
        print('shape of trainx =', trainx.shape)
        print('shape of trainy =', trainy.shape)
        label_DGW = np.repeat(trainy, 32, axis=0)  # Converting the labels into total number of time stamp
        print(label_DGW.shape)
        label_DGW = label_DGW.reshape(label_DGW.shape[0], 1)
        data_DGW = trainx.reshape(trainx.shape[0] * trainx.shape[1], trainx.shape[2])
        data_DGW = np.concatenate((data_DGW, label_DGW), axis=1)                    # Adding the labels to the syntetic data generated from DGW
        data_aug = np.concatenate((data, data_DGW), axis=0)                         # Adding the original data to the syntetic data generated from conditional GAN
        print('Shape of time warped data:', data_aug.shape)
               
    elif method == 'dicriminative_guided_warp':
        trainxs1r1, trainys1r1 = DGW(s1r1)
        trainxs1r2, trainys1r2 = DGW(s1r2)
        trainxs1r3, trainys1r3 = DGW(s1r3)
        trainxs1_drill, trainys1_drill = DGW(s1_drill)
        trainxs1r4, trainys1r4 = DGW(s1r4)
        trainxs1r5, trainys1r5 = DGW(s1r5)
        trainxs2r1, trainys2r1 = DGW(s2r1)
        trainxs2r2, trainys2r2 = DGW(s2r2)
        trainxs2_drill, trainys2_drill = DGW(s2_drill)
        trainxs3r1, trainys3r1 = DGW(s3r1)
        trainxs3r2, trainys3r2 = DGW(s3r2)
        trainxs3_drill, trainys3_drill = DGW(s3_drill)
        trainx = np.concatenate((trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
                 trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)
        trainy = np.concatenate((trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
                 trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)
        print('shape of trainx =', trainx.shape)
        print('shape of trainy =', trainy.shape)
        label_DGW = np.repeat(trainy, 32, axis=0)  # Converting the labels into total number of time stamp
        print(label_DGW.shape)
        label_DGW = label_DGW.reshape(label_DGW.shape[0], 1)
        data_DGW = trainx.reshape(trainx.shape[0] * trainx.shape[1], trainx.shape[2])
        data_DGW = np.concatenate((data_DGW, label_DGW), axis=1)                    # Adding the labels to the syntetic data generated from DGW
        data_aug = np.concatenate((data, data_DGW), axis=0)                         # Adding the original data to the syntetic data generated from conditional GAN
        print('Shape of time warped data:', data_aug.shape)
        
    elif method == 'time_warp':
        trainxs1r1, trainys1r1 = TW(s1r1)
        trainxs1r2, trainys1r2 = TW(s1r2)
        trainxs1r3, trainys1r3 = TW(s1r3)
        trainxs1_drill, trainys1_drill = TW(s1_drill)
        trainxs1r4, trainys1r4 = TW(s1r4)
        trainxs1r5, trainys1r5 = TW(s1r5)
        trainxs2r1, trainys2r1 = TW(s2r1)
        trainxs2r2, trainys2r2 = TW(s2r2)
        trainxs2_drill, trainys2_drill = TW(s2_drill)
        trainxs3r1, trainys3r1 = TW(s3r1)
        trainxs3r2, trainys3r2 = TW(s3r2)
        trainxs3_drill, trainys3_drill = TW(s3_drill)
        trainx = np.concatenate((trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
                 trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)
        trainy = np.concatenate((trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
                 trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)
        print('shape of trainx =', trainx.shape)
        print('shape of trainy =', trainy.shape)
        label_TW = np.repeat(trainy, 32, axis=0)  # Converting the labels into total number of time stamp
        print(label_TW.shape)
        label_TW = label_TW.reshape(label_TW.shape[0], 1)
        data_TW = trainx.reshape(trainx.shape[0] * trainx.shape[1], trainx.shape[2])
        data_TW = np.concatenate((data_TW, label_TW), axis=1)                    # Adding the labels to the syntetic data generated from TW
        data_aug = np.concatenate((data, data_TW), axis=0)                         # Adding the original data to the syntetic data generated from conditional GAN
        print('Shape of time warped data:', data_aug.shape)
    
    return data_aug


method = input('Mention the Method Name (SMOTE, TSAUG, GAN, time_warp, discriminative_guided_warp, random_guided_warp):')     # Enter the name of the Method that you want to test
data_aug = aug_method(method)
# Converting the Numpy data to Pandas Dataframe 
df_method = pd.DataFrame(data_aug)
df_method = column_notation(df_method) # Addiitng the Notation of column values.


# Grouping by the PD dataframe
gk_method = df_method.groupby('Activity_Label')
df_No_activity_method = gk_method.get_group(0)
df_open_door_1_method = gk_method.get_group(1)
df_open_door_2_method = gk_method.get_group(2)
df_close_door_1_method = gk_method.get_group(3)
df_close_door_2_method = gk_method.get_group(4)
df_open_fridge_method = gk_method.get_group(5)
df_close_fridge_method  = gk_method.get_group(6)
df_open_dishwasher_method = gk_method.get_group(7)
df_close_dishwasher_method = gk_method.get_group(8)
df_open_dawer_1_method = gk_method.get_group(9)
df_close_dawer_1_method = gk_method.get_group(10)
df_open_dawer_2_method = gk_method.get_group(11)
df_close_dawer_2_method = gk_method.get_group(12)
df_open_dawer_3_method = gk_method.get_group(13)
df_close_dawer_3_method = gk_method.get_group(14)
df_clean_table_method = gk_method.get_group(15)
df_drink_from_cup_method = gk_method.get_group(16)
df_toggle_switch_method = gk_method.get_group(17)
# Seperating the data and lavels
No_activity_method, No_activity_label_method = df_No_activity_method.iloc[:, :-1], df_No_activity_method.iloc[:, [-1]]
open_door_1_method, open_door_1_label_method = df_open_door_1_method.iloc[:, :-1], df_open_door_1_method.iloc[:, [-1]]
open_door_2_method, open_door_2_label_method = df_open_door_2_method.iloc[:, :-1], df_open_door_2_method.iloc[:, [-1]]
close_door_1_method, close_door_1_label_method = df_close_door_1_method.iloc[:, :-1], df_close_door_1_method.iloc[:, [-1]]
close_door_2_method, close_door_2_label_method = df_close_door_2_method.iloc[:, :-1], df_close_door_2_method.iloc[:, [-1]]
open_fridge_method, open_fridge_label_method = df_open_fridge_method.iloc[:, :-1], df_open_fridge_method.iloc[:, [-1]]
close_fridge_method, close_fridge_label_method = df_close_fridge_method.iloc[:, :-1], df_close_fridge_method.iloc[:, [-1]]
open_dishwasher_method, open_dishwasher_label_method = df_open_dishwasher_method.iloc[:, :-1], df_open_dishwasher_method.iloc[:, [-1]]
close_dishwasher_method, close_dishwasher_label_method = df_close_dishwasher_method.iloc[:, :-1], df_close_dishwasher_method.iloc[:, [-1]]
open_dawer_1_method, open_dawer_1_label_method = df_open_dawer_1_method.iloc[:, :-1], df_open_dawer_1_method.iloc[:, [-1]]
close_dawer_1_method, close_dawer_1_label_method = df_close_dawer_1_method.iloc[:, :-1], df_close_dawer_1_method.iloc[:, [-1]]
open_dawer_2_method, open_dawer_2_label_method = df_open_dawer_2_method.iloc[:, :-1], df_open_dawer_2_method.iloc[:, [-1]]
close_dawer_2_method, close_dawer_2_label_method = df_close_dawer_2_method.iloc[:, :-1], df_close_dawer_2_method.iloc[:, [-1]]
open_dawer_3_method, open_dawer_3_label_method = df_open_dawer_3_method.iloc[:, :-1], df_open_dawer_3_method.iloc[:, [-1]]
close_dawer_3_method, close_dawer_3_label_method = df_close_dawer_3_method.iloc[:, :-1], df_close_dawer_3_method.iloc[:, [-1]]
clean_table_method, clean_table_label_method = df_clean_table_method.iloc[:, :-1], df_clean_table_method.iloc[:, [-1]]
drink_from_cup_method, df_drink_from_cup_label_method = df_drink_from_cup_method.iloc[:, :-1], df_drink_from_cup_method.iloc[:, [-1]]
toggle_switch_method, toggle_switch_label_method = df_toggle_switch_method.iloc[:, :-1], df_toggle_switch_method.iloc[:, [-1]]
    
# Converting the Data into numpy array to find the FID score
No_activity_method = No_activity_method.to_numpy()
open_door_1_method = open_door_1_method.to_numpy()
open_door_2_method = open_door_2_method.to_numpy()
close_door_1_method = close_door_1_method.to_numpy()
close_door_2_method = close_door_2_method.to_numpy()
open_fridge_method = open_fridge_method.to_numpy()
close_fridge_method = close_fridge_method.to_numpy()
open_dishwasher_method = open_dishwasher_method.to_numpy()
close_dishwasher_method = close_dishwasher_method.to_numpy()
open_dawer_1_method = open_dawer_1_method.to_numpy()
close_dawer_1_method = close_dawer_1_method.to_numpy()
open_dawer_2_method = open_dawer_2_method.to_numpy()
close_dawer_2_method = close_dawer_2_method.to_numpy()
open_dawer_3_method = open_dawer_3_method.to_numpy
close_dawer_3_method = close_dawer_3_method.to_numpy()
clean_table_method = clean_table_method.to_numpy()
drink_from_cup_method = drink_from_cup_method.to_numpy()
toggle_switch_method = toggle_switch_method.to_numpy()
#------------------------------------------Calculating FID score-------------------------------------------------------------------------------------
# # fid between act1 and act2
fid = calculate_fid(open_door_1, open_door_1_method)
print("FID open_door 1 With"+" "+ method +": %.3f" % fid)
fid = calculate_fid(open_door_2, open_door_2_method)
print("FID open_door 2 With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(close_door_1, close_door_1_method)
print("FID close_door 1 With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(close_door_2, close_door_2_method)
print("FID close_door 2 With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(open_fridge, open_fridge_method)
print("FID open_Fridge With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(close_fridge, close_fridge_method)
print("FID close_fridge With "+" "+ method +": %.3f" % fid)
fid= calculate_fid(open_dishwasher, open_dishwasher_method)
print("FID open_dishwasher With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(close_dishwasher, close_dishwasher_method)
print("FID close_dishwasher With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(open_dawer_1, open_dawer_1_method)
print("FID open_dawer 1 With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(close_dawer_1, close_dawer_1_method)
print("FID close_dawer 1 With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(open_dawer_2, open_dawer_2_method)
print("FID open_dawer 2 With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(close_dawer_2, close_dawer_2_method)
print("FID close_dawer 2 With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(close_dawer_3, close_dawer_3_method)
print("FID close_dawer 3 With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(clean_table, clean_table_method)
print("FID clean table With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(drink_from_cup, drink_from_cup_method)
print("FID Drink from cup With "+" "+ method +": %.3f" % fid)
fid = calculate_fid(toggle_switch, toggle_switch_method)
print("FID toggle switch With "+" "+ method +": %.3f" % fid)