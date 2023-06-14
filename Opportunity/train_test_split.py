def train_test_split():
    global trainx, trainy, testx, testy, valx, valy
    import pandas as pd
    import numpy as np
    import load_data
    import windowed_numpy_3Darray
    import seg_SMOTE
    import seg_TSAUG
    import synthetic_data_generator

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
    activities = {
        'Old label': [0, 406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511,
                      406508, 404508, 408512, 407521, 405506],
        'New label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        'Activity': ['No activity', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'close Door 2', 'Open fridge',
                     'Close fridge',
                     'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2',
                     'Close Drawer 2',
                     'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup', 'Toggle Switch']}
    activities = pd.DataFrame(activities)
    print(activities)
    ACTIVITY = ['No activity', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'close Door 2', 'Open fridge',
                'Close fridge',
                'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2',
                'Close Drawer 2',
                'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup', 'Toggle Switch']

# ----------------------- Asking Which Method Do you want to use.

    family = input('Enter the Augmentation Family NAME:')

# --------------------------Applying the SMOTE to Generate the syntetic data--------------------------------------------
    if family == 'SMOTE':
        s1r1 = seg_SMOTE.seg_SMOTE(s1r1)
        s1r2 = seg_SMOTE.seg_SMOTE(s1r2)
        s1r3 = seg_SMOTE.seg_SMOTE(s1r3)
        s1r4 = seg_SMOTE.seg_SMOTE(s1r4)
        s1r5 = seg_SMOTE.seg_SMOTE(s1r5)
        s1_drill = seg_SMOTE.seg_SMOTE(s1_drill)
        s2r1 = seg_SMOTE.seg_SMOTE(s2r1)
        s2r2 = seg_SMOTE.seg_SMOTE(s2r2)
        s2r3 = seg_SMOTE.seg_SMOTE(s2r3)
        s2r4 = seg_SMOTE.seg_SMOTE(s2r4)
        s2r5 = seg_SMOTE.seg_SMOTE(s2r5)
        s2_drill = seg_SMOTE.seg_SMOTE(s2_drill)
        s3r1 = seg_SMOTE.seg_SMOTE(s3r1)
        s3r2 = seg_SMOTE.seg_SMOTE(s3r2)
        s3r3 = seg_SMOTE.seg_SMOTE(s3r3)
        s3r4 = seg_SMOTE.seg_SMOTE(s3r4)
        s3r5 = seg_SMOTE.seg_SMOTE(s3r5)
        s3_drill = seg_SMOTE.seg_SMOTE(s3_drill)
        s4r1 = seg_SMOTE.seg_SMOTE(s4r1)
        s4r2 = seg_SMOTE.seg_SMOTE(s4r2)
        s4r3 = seg_SMOTE.seg_SMOTE(s4r3)
        s4r4 = seg_SMOTE.seg_SMOTE(s4r4)
        s4r5 = seg_SMOTE.seg_SMOTE(s4r5)
        s4_drill = seg_SMOTE.seg_SMOTE(s4_drill)

        # model_paper = input('Enter the Model Paper:')
        model_paper = 'ROGGEN'

        if model_paper == 'FREDERIC':
            # ----------------- Frederic's paper Stratagey-------------------------------------------------------------------
            # Converting 2d array in to 3D array with slided window algorithm where the overlap is 50% and window size is fixed 64 (2 sec).
            trainxs1r1, trainys1r1 = windowed_numpy_3Darray.slided_numpy_array(s1r1)
            trainxs1r2, trainys1r2 = windowed_numpy_3Darray.slided_numpy_array(s1r2)
            trainxs1r3, trainys1r3 = windowed_numpy_3Darray.slided_numpy_array(s1r3)
            trinxs1_drill, trainys1_drill = windowed_numpy_3Darray.slided_numpy_array(s1_drill)
            testxs1r4, testys1r4 = windowed_numpy_3Darray.slided_numpy_array(s1r4)
            testxs1r5, testys1r5 = windowed_numpy_3Darray.slided_numpy_array(s1r5)
            trainxs2r1, trainys2r1 = windowed_numpy_3Darray.slided_numpy_array(s2r1)
            trainxs2r2, trainys2r2 = windowed_numpy_3Darray.slided_numpy_array(s2r2)
            trainxs2r3, trainys2r3 = windowed_numpy_3Darray.slided_numpy_array(s2r3)
            trinxs2_drill, trainys2_drill = windowed_numpy_3Darray.slided_numpy_array(s2_drill)
            testxs2r4, testys2r4 = windowed_numpy_3Darray.slided_numpy_array(s2r4)
            testxs2r5, testys2r5 = windowed_numpy_3Darray.slided_numpy_array(s2r5)
            trainxs3r1, trainys3r1 = windowed_numpy_3Darray.slided_numpy_array(s3r1)
            trainxs3r2, trainys3r2 = windowed_numpy_3Darray.slided_numpy_array(s3r2)
            trainxs3r3, trainys3r3 = windowed_numpy_3Darray.slided_numpy_array(s3r3)
            trinxs3_drill, trainys3_drill = windowed_numpy_3Darray.slided_numpy_array(s3_drill)
            testxs3r4, testys3r4 = windowed_numpy_3Darray.slided_numpy_array(s3r4)
            testxs3r5, testys3r5 = windowed_numpy_3Darray.slided_numpy_array(s3r5)
            trainxs4r1, trainys4r1 = windowed_numpy_3Darray.slided_numpy_array(s4r1)
            trainxs4r2, trainys4r2 = windowed_numpy_3Darray.slided_numpy_array(s4r2)
            trainxs4r3, trainys4r3 = windowed_numpy_3Darray.slided_numpy_array(s4r3)
            trinxs4_drill, trainys4_drill = windowed_numpy_3Darray.slided_numpy_array(s4_drill)
            testxs4r4, testys4r4 = windowed_numpy_3Darray.slided_numpy_array(s4r4)
            testxs4r5, testys4r5 = windowed_numpy_3Darray.slided_numpy_array(s4r5)
            trainx = np.concatenate(
                (trainxs1r1, trainxs1r2, trainxs1r3, trinxs1_drill, trainxs2r1, trainxs2r2, trainxs2r3,
                 trinxs2_drill, trainxs3r1, trainxs3r2, trainxs3r3, trinxs3_drill, trainxs4r1, trainxs4r2, trainxs4r3,
                 trinxs4_drill), axis=0)
            trainy = np.concatenate((trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys2r1, trainys2r2,
                                     trainys2r3, trainys2_drill, trainys3r1,
                                     trainys3r2, trainys3r3, trainys3_drill, trainys4r1, trainys4r2, trainys4r3,
                                     trainys4_drill), axis=0)
            testx = np.concatenate(
                (testxs1r4, testxs1r5, testxs2r4, testxs2r5, testxs3r4, testxs3r5, testxs4r4, testxs4r5), axis=0)
            testy = np.concatenate(
                (testys1r4, testys1r5, testys2r4, testys2r5, testys3r4, testys3r5, testys4r4, testys4r5), axis=0)
            valx = np.concatenate((trainxs2r3, trainxs3r3), axis=0)

            valy = np.concatenate((trainys2r3, trainys3r3), axis=0)

            # ---------------------------------Rogen's Paper Stratagey-----------------------------------------------------------------------------
            # Converting 2d array in to 3D array with slided window algorithm where the overlap is 50% and window size is fixed 32 (1 sec).
        elif model_paper == 'ROGGEN':
            trainxs1r1, trainys1r1 = windowed_numpy_3Darray.slided_numpy_array(s1r1)
            trainxs1r2, trainys1r2 = windowed_numpy_3Darray.slided_numpy_array(s1r2)
            trainxs1r3, trainys1r3 = windowed_numpy_3Darray.slided_numpy_array(s1r3)
            trainxs1_drill, trainys1_drill = windowed_numpy_3Darray.slided_numpy_array(s1_drill)
            trainxs1r4, trainys1r4 = windowed_numpy_3Darray.slided_numpy_array(s1r4)
            trainxs1r5, trainys1r5 = windowed_numpy_3Darray.slided_numpy_array(s1r5)
            trainxs2r1, trainys2r1 = windowed_numpy_3Darray.slided_numpy_array(s2r1)
            trainxs2r2, trainys2r2 = windowed_numpy_3Darray.slided_numpy_array(s2r2)
            trainxs2_drill, trainys2_drill = windowed_numpy_3Darray.slided_numpy_array(s2_drill)
            trainxs3r1, trainys3r1 = windowed_numpy_3Darray.slided_numpy_array(s3r1)
            trainxs3r2, trainys3r2 = windowed_numpy_3Darray.slided_numpy_array(s3r2)
            trainxs3_drill, trainys3_drill = windowed_numpy_3Darray.slided_numpy_array(s3_drill)
            valxs2r3, valys2r3 = windowed_numpy_3Darray.slided_numpy_array(s2r3)
            valxs3r3, valys3r3 = windowed_numpy_3Darray.slided_numpy_array(s3r3)
            testxs2r4, testys2r4 = windowed_numpy_3Darray.slided_numpy_array(s2r4)
            testxs2r5, testys2r5 = windowed_numpy_3Darray.slided_numpy_array(s2r5)
            testxs3r4, testys3r4 = windowed_numpy_3Darray.slided_numpy_array(s3r4)
            testxs3r5, testys3r5 = windowed_numpy_3Darray.slided_numpy_array(s3r5)

            trainx = np.concatenate(
                (trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
                 trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)

            trainy = np.concatenate(
                (trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
                 trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)

            valx = np.concatenate((valxs2r3, valxs3r3), axis=0)

            valy = np.concatenate((valys2r3, valys3r3), axis=0)

            testx = np.concatenate((testxs2r4, testxs2r5, testxs3r4, testxs3r5), axis=0)
            testy = np.concatenate((testys2r4, testys2r5, testys3r4, testys3r5), axis=0)
            print('shape of trainx =', trainx.shape)
            print('shape of trainy =', trainy.shape)
            print('Shape of testx =', testx.shape)
            print('Shape of testy =', testy.shape)
            print('Shape of Valx=', valx.shape)
            print('Shape of valy=', valy.shape)

# ----------------------- Applying Basic Approach to generate syntetic data---------------------------------------------
    elif family == 'TSAUG':
        #model_paper = input('Enter The Model Paper:')
        model_paper = 'ROGGEN'
        if model_paper == 'FREDERIC':
            trainxs1r1, trainys1r1 = seg_TSAUG.seg_TSAUG(s1r1)
            trainxs1r2, trainys1r2 = seg_TSAUG.seg_TSAUG(s1r2)
            trainxs1r3, trainys1r3 = seg_TSAUG.seg_TSAUG(s1r3)
            trinxs1_drill, trainys1_drill = seg_TSAUG.seg_TSAUG(s1_drill)

            testxs1r4, testys1r4 = windowed_numpy_3Darray.slided_numpy_array(s1r4)
            testxs1r5, testys1r5 = windowed_numpy_3Darray.slided_numpy_array(s1r5)

            trainxs2r1, trainys2r1 = seg_TSAUG.seg_TSAUG(s2r1)
            trainxs2r2, trainys2r2 = seg_TSAUG.seg_TSAUG(s2r2)
            trainxs2r3, trainys2r3 = seg_TSAUG.seg_TSAUG(s2r3)
            trinxs2_drill, trainys2_drill = seg_TSAUG.seg_TSAUG(s2_drill)

            testxs2r4, testys2r4 = windowed_numpy_3Darray.slided_numpy_array(s2r4)
            testxs2r5, testys2r5 = windowed_numpy_3Darray.slided_numpy_array(s2r5)

            trainxs3r1, trainys3r1 = seg_TSAUG.seg_TSAUG(s3r1)
            trainxs3r2, trainys3r2 = seg_TSAUG.seg_TSAUG(s3r2)
            trainxs3r3, trainys3r3 = seg_TSAUG.seg_TSAUG(s3r3)
            trinxs3_drill, trainys3_drill = seg_TSAUG.seg_TSAUG(s3_drill)

            testxs3r4, testys3r4 = windowed_numpy_3Darray.slided_numpy_array(s3r4)
            testxs3r5, testys3r5 = windowed_numpy_3Darray.slided_numpy_array(s3r5)

            trainxs4r1, trainys4r1 = seg_TSAUG.seg_TSAUG(s4r1)
            trainxs4r2, trainys4r2 = seg_TSAUG.seg_TSAUG(s4r2)
            trainxs4r3, trainys4r3 = seg_TSAUG.seg_TSAUG(s4r3)
            trinxs4_drill, trainys4_drill = seg_TSAUG.seg_TSAUG(s4_drill)

            testxs4r4, testys4r4 = windowed_numpy_3Darray.slided_numpy_array(s4r4)
            testxs4r5, testys4r5 = windowed_numpy_3Darray.slided_numpy_array(s4r5)

            trainx = np.concatenate(
                (trainxs1r1, trainxs1r2, trainxs1r3, trinxs1_drill, trainxs2r1, trainxs2r2, trainxs2r3,
                 trinxs2_drill, trainxs3r1, trainxs3r2, trainxs3r3, trinxs3_drill, trainxs4r1, trainxs4r2, trainxs4r3,
                 trinxs4_drill), axis=0)
            trainy = np.concatenate((trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys2r1, trainys2r2,
                                     trainys2r3, trainys2_drill, trainys3r1,
                                     trainys3r2, trainys3r3, trainys3_drill, trainys4r1, trainys4r2, trainys4r3,
                                     trainys4_drill), axis=0)
            testx = np.concatenate(
                (testxs1r4, testxs1r5, testxs2r4, testxs2r5, testxs3r4, testxs3r5, testxs4r4, testxs4r5), axis=0)
            testy = np.concatenate(
                (testys1r4, testys1r5, testys2r4, testys2r5, testys3r4, testys3r5, testys4r4, testys4r5), axis=0)

            valx = testxs1r4

            valy = testys1r4


        elif model_paper == 'ROGGEN':
            trainxs1r1, trainys1r1 = seg_TSAUG.seg_TSAUG(s1r1)
            trainxs1r2, trainys1r2 = seg_TSAUG.seg_TSAUG(s1r2)
            trainxs1r3, trainys1r3 = seg_TSAUG.seg_TSAUG(s1r3)
            trainxs1_drill, trainys1_drill = seg_TSAUG.seg_TSAUG(s1_drill)
            trainxs1r4, trainys1r4 = seg_TSAUG.seg_TSAUG(s1r4)
            trainxs1r5, trainys1r5 = seg_TSAUG.seg_TSAUG(s1r5)
            trainxs2r1, trainys2r1 = seg_TSAUG.seg_TSAUG(s2r1)
            trainxs2r2, trainys2r2 = seg_TSAUG.seg_TSAUG(s2r2)
            trainxs2_drill, trainys2_drill = seg_TSAUG.seg_TSAUG(s2_drill)
            trainxs3r1, trainys3r1 = seg_TSAUG.seg_TSAUG(s3r1)
            trainxs3r2, trainys3r2 = seg_TSAUG.seg_TSAUG(s3r2)
            trainxs3_drill, trainys3_drill = seg_TSAUG.seg_TSAUG(s3_drill)

            valxs2r3, valys2r3 = windowed_numpy_3Darray.slided_numpy_array(s2r3)
            valxs3r3, valys3r3 = windowed_numpy_3Darray.slided_numpy_array(s3r3)

            testxs2r4, testys2r4 = windowed_numpy_3Darray.slided_numpy_array(s2r4)
            testxs2r5, testys2r5 = windowed_numpy_3Darray.slided_numpy_array(s2r5)
            testxs3r4, testys3r4 = windowed_numpy_3Darray.slided_numpy_array(s3r4)
            testxs3r5, testys3r5 = windowed_numpy_3Darray.slided_numpy_array(s3r5)

            trainx = np.concatenate(
                (trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
                 trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)

            trainy = np.concatenate(
                (trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
                 trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)

            valx = np.concatenate((valxs2r3, valxs3r3), axis=0)

            valy = np.concatenate((valys2r3, valys3r3), axis=0)

            testx = np.concatenate((testxs2r4, testxs2r5, testxs3r4, testxs3r5), axis=0)
            testy = np.concatenate((testys2r4, testys2r5, testys3r4, testys3r5), axis=0)

            print('shape of trainx =', trainx.shape)
            print('shape of trainy =', trainy.shape)
            print('Shape of testx =', testx.shape)
            print('Shape of testy =', testy.shape)
            print('Shape of Valx=', valx.shape)
            print('Shape of valy=', valy.shape)
    elif family == 'GAN':
        # model_paper = input('Enter The Model Paper:')
        model_paper = 'ROGGEN'
        if model_paper == 'FREDERIC':
            # ----------------- Frederic's paper Stratagey-------------------------------------------------------------------
            # Converting 2d array in to 3D array with slided window algorithm where the overlap is 50% and window size is fixed 64 (2 sec).
            trainxs1r1, trainys1r1 = windowed_numpy_3Darray.slided_numpy_array(s1r1)
            trainxs1r2, trainys1r2 = windowed_numpy_3Darray.slided_numpy_array(s1r2)
            trainxs1r3, trainys1r3 = windowed_numpy_3Darray.slided_numpy_array(s1r3)
            trinxs1_drill, trainys1_drill = windowed_numpy_3Darray.slided_numpy_array(s1_drill)
            testxs1r4, testys1r4 = windowed_numpy_3Darray.slided_numpy_array(s1r4)
            testxs1r5, testys1r5 = windowed_numpy_3Darray.slided_numpy_array(s1r5)
            trainxs2r1, trainys2r1 = windowed_numpy_3Darray.slided_numpy_array(s2r1)
            trainxs2r2, trainys2r2 = windowed_numpy_3Darray.slided_numpy_array(s2r2)
            trainxs2r3, trainys2r3 = windowed_numpy_3Darray.slided_numpy_array(s2r3)
            trinxs2_drill, trainys2_drill = windowed_numpy_3Darray.slided_numpy_array(s2_drill)
            testxs2r4, testys2r4 = windowed_numpy_3Darray.slided_numpy_array(s2r4)
            testxs2r5, testys2r5 = windowed_numpy_3Darray.slided_numpy_array(s2r5)
            trainxs3r1, trainys3r1 = windowed_numpy_3Darray.slided_numpy_array(s3r1)
            trainxs3r2, trainys3r2 = windowed_numpy_3Darray.slided_numpy_array(s3r2)
            trainxs3r3, trainys3r3 = windowed_numpy_3Darray.slided_numpy_array(s3r3)
            trinxs3_drill, trainys3_drill = windowed_numpy_3Darray.slided_numpy_array(s3_drill)
            testxs3r4, testys3r4 = windowed_numpy_3Darray.slided_numpy_array(s3r4)
            testxs3r5, testys3r5 = windowed_numpy_3Darray.slided_numpy_array(s3r5)
            trainxs4r1, trainys4r1 = windowed_numpy_3Darray.slided_numpy_array(s4r1)
            trainxs4r2, trainys4r2 = windowed_numpy_3Darray.slided_numpy_array(s4r2)
            trainxs4r3, trainys4r3 = windowed_numpy_3Darray.slided_numpy_array(s4r3)
            trinxs4_drill, trainys4_drill = windowed_numpy_3Darray.slided_numpy_array(s4_drill)
            testxs4r4, testys4r4 = windowed_numpy_3Darray.slided_numpy_array(s4r4)
            testxs4r5, testys4r5 = windowed_numpy_3Darray.slided_numpy_array(s4r5)
            trainx = np.concatenate(
                (trainxs1r1, trainxs1r2, trainxs1r3, trinxs1_drill, trainxs2r1, trainxs2r2, trainxs2r3,
                 trinxs2_drill, trainxs3r1, trainxs3r2, trainxs3r3, trinxs3_drill, trainxs4r1, trainxs4r2, trainxs4r3,
                 trinxs4_drill), axis=0)
            trainy = np.concatenate((trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys2r1, trainys2r2,
                                     trainys2r3, trainys2_drill, trainys3r1,
                                     trainys3r2, trainys3r3, trainys3_drill, trainys4r1, trainys4r2, trainys4r3,
                                     trainys4_drill), axis=0)
            testx = np.concatenate(
                (testxs1r4, testxs1r5, testxs2r4, testxs2r5, testxs3r4, testxs3r5, testxs4r4, testxs4r5), axis=0)
            testy = np.concatenate(
                (testys1r4, testys1r5, testys2r4, testys2r5, testys3r4, testys3r5, testys4r4, testys4r5), axis=0)
            valx = np.concatenate((trainxs2r3, trainxs3r3), axis=0)

            valy = np.concatenate((trainys2r3, trainys3r3), axis=0)


        elif model_paper == 'ROGGEN':
            trainxs1r1, trainys1r1 = windowed_numpy_3Darray.slided_numpy_array(s1r1)
            trainxs1r2, trainys1r2 = windowed_numpy_3Darray.slided_numpy_array(s1r2)
            trainxs1r3, trainys1r3 = windowed_numpy_3Darray.slided_numpy_array(s1r3)
            trainxs1_drill, trainys1_drill = windowed_numpy_3Darray.slided_numpy_array(s1_drill)
            trainxs1r4, trainys1r4 = windowed_numpy_3Darray.slided_numpy_array(s1r4)
            trainxs1r5, trainys1r5 = windowed_numpy_3Darray.slided_numpy_array(s1r5)
            trainxs2r1, trainys2r1 = windowed_numpy_3Darray.slided_numpy_array(s2r1)
            trainxs2r2, trainys2r2 = windowed_numpy_3Darray.slided_numpy_array(s2r2)
            trainxs2_drill, trainys2_drill = windowed_numpy_3Darray.slided_numpy_array(s2_drill)
            trainxs3r1, trainys3r1 = windowed_numpy_3Darray.slided_numpy_array(s3r1)
            trainxs3r2, trainys3r2 = windowed_numpy_3Darray.slided_numpy_array(s3r2)
            trainxs3_drill, trainys3_drill = windowed_numpy_3Darray.slided_numpy_array(s3_drill)
            valxs2r3, valys2r3 = windowed_numpy_3Darray.slided_numpy_array(s2r3)
            valxs3r3, valys3r3 = windowed_numpy_3Darray.slided_numpy_array(s3r3)
            testxs2r4, testys2r4 = windowed_numpy_3Darray.slided_numpy_array(s2r4)
            testxs2r5, testys2r5 = windowed_numpy_3Darray.slided_numpy_array(s2r5)
            testxs3r4, testys3r4 = windowed_numpy_3Darray.slided_numpy_array(s3r4)
            testxs3r5, testys3r5 = windowed_numpy_3Darray.slided_numpy_array(s3r5)

            # Generate the  synthetic data from the Generator
            X_synthetic, Y_synthetic = synthetic_data_generator.GAN_generator()

            trainx = np.concatenate((trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
                 trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill, X_synthetic), axis=0)

            trainy = np.concatenate((trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
                 trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill, Y_synthetic), axis=0)

            valx = np.concatenate((valxs2r3, valxs3r3), axis=0)

            valy = np.concatenate((valys2r3, valys3r3), axis=0)

            testx = np.concatenate((testxs2r4, testxs2r5, testxs3r4, testxs3r5), axis=0)
            testy = np.concatenate((testys2r4, testys2r5, testys3r4, testys3r5), axis=0)
            print('shape of trainx =', trainx.shape)
            print('shape of trainy =', trainy.shape)
            print('Shape of testx =', testx.shape)
            print('Shape of testy =', testy.shape)
            print('Shape of Valx=', valx.shape)
            print('Shape of valy=', valy.shape)

    elif family == 'No_Aug':
        model_paper = 'ROGGEN'

        if model_paper == 'FREDERIC':
            # ----------------- Frederic's paper Stratagey-------------------------------------------------------------------
            # Converting 2d array in to 3D array with slided window algorithm where the overlap is 50% and window size is fixed 64 (2 sec).
            trainxs1r1, trainys1r1 = windowed_numpy_3Darray.slided_numpy_array(s1r1)
            trainxs1r2, trainys1r2 = windowed_numpy_3Darray.slided_numpy_array(s1r2)
            trainxs1r3, trainys1r3 = windowed_numpy_3Darray.slided_numpy_array(s1r3)
            trinxs1_drill, trainys1_drill = windowed_numpy_3Darray.slided_numpy_array(s1_drill)
            testxs1r4, testys1r4 = windowed_numpy_3Darray.slided_numpy_array(s1r4)
            testxs1r5, testys1r5 = windowed_numpy_3Darray.slided_numpy_array(s1r5)
            trainxs2r1, trainys2r1 = windowed_numpy_3Darray.slided_numpy_array(s2r1)
            trainxs2r2, trainys2r2 = windowed_numpy_3Darray.slided_numpy_array(s2r2)
            trainxs2r3, trainys2r3 = windowed_numpy_3Darray.slided_numpy_array(s2r3)
            trinxs2_drill, trainys2_drill = windowed_numpy_3Darray.slided_numpy_array(s2_drill)
            testxs2r4, testys2r4 = windowed_numpy_3Darray.slided_numpy_array(s2r4)
            testxs2r5, testys2r5 = windowed_numpy_3Darray.slided_numpy_array(s2r5)
            trainxs3r1, trainys3r1 = windowed_numpy_3Darray.slided_numpy_array(s3r1)
            trainxs3r2, trainys3r2 = windowed_numpy_3Darray.slided_numpy_array(s3r2)
            trainxs3r3, trainys3r3 = windowed_numpy_3Darray.slided_numpy_array(s3r3)
            trinxs3_drill, trainys3_drill = windowed_numpy_3Darray.slided_numpy_array(s3_drill)
            testxs3r4, testys3r4 = windowed_numpy_3Darray.slided_numpy_array(s3r4)
            testxs3r5, testys3r5 = windowed_numpy_3Darray.slided_numpy_array(s3r5)
            trainxs4r1, trainys4r1 = windowed_numpy_3Darray.slided_numpy_array(s4r1)
            trainxs4r2, trainys4r2 = windowed_numpy_3Darray.slided_numpy_array(s4r2)
            trainxs4r3, trainys4r3 = windowed_numpy_3Darray.slided_numpy_array(s4r3)
            trinxs4_drill, trainys4_drill = windowed_numpy_3Darray.slided_numpy_array(s4_drill)
            testxs4r4, testys4r4 = windowed_numpy_3Darray.slided_numpy_array(s4r4)
            testxs4r5, testys4r5 = windowed_numpy_3Darray.slided_numpy_array(s4r5)
            trainx = np.concatenate(
                (trainxs1r1, trainxs1r2, trainxs1r3, trinxs1_drill, trainxs2r1, trainxs2r2, trainxs2r3,
                 trinxs2_drill, trainxs3r1, trainxs3r2, trainxs3r3, trinxs3_drill, trainxs4r1, trainxs4r2, trainxs4r3,
                 trinxs4_drill), axis=0)
            trainy = np.concatenate((trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys2r1, trainys2r2,
                                     trainys2r3, trainys2_drill, trainys3r1,
                                     trainys3r2, trainys3r3, trainys3_drill, trainys4r1, trainys4r2, trainys4r3,
                                     trainys4_drill), axis=0)
            testx = np.concatenate(
                (testxs1r4, testxs1r5, testxs2r4, testxs2r5, testxs3r4, testxs3r5, testxs4r4, testxs4r5), axis=0)
            testy = np.concatenate(
                (testys1r4, testys1r5, testys2r4, testys2r5, testys3r4, testys3r5, testys4r4, testys4r5), axis=0)
            valx = np.concatenate((trainxs2r3, trainxs3r3), axis=0)

            valy = np.concatenate((trainys2r3, trainys3r3), axis=0)

            # ---------------------------------Rogen's Paper Stratagey-----------------------------------------------------------------------------
            # Converting 2d array in to 3D array with slided window algorithm where the overlap is 50% and window size is fixed 32 (1 sec).
        elif model_paper == 'ROGGEN':
            trainxs1r1, trainys1r1 = windowed_numpy_3Darray.slided_numpy_array(s1r1)
            trainxs1r2, trainys1r2 = windowed_numpy_3Darray.slided_numpy_array(s1r2)
            trainxs1r3, trainys1r3 = windowed_numpy_3Darray.slided_numpy_array(s1r3)
            trainxs1_drill, trainys1_drill = windowed_numpy_3Darray.slided_numpy_array(s1_drill)
            trainxs1r4, trainys1r4 = windowed_numpy_3Darray.slided_numpy_array(s1r4)
            trainxs1r5, trainys1r5 = windowed_numpy_3Darray.slided_numpy_array(s1r5)
            trainxs2r1, trainys2r1 = windowed_numpy_3Darray.slided_numpy_array(s2r1)
            trainxs2r2, trainys2r2 = windowed_numpy_3Darray.slided_numpy_array(s2r2)
            trainxs2_drill, trainys2_drill = windowed_numpy_3Darray.slided_numpy_array(s2_drill)
            trainxs3r1, trainys3r1 = windowed_numpy_3Darray.slided_numpy_array(s3r1)
            trainxs3r2, trainys3r2 = windowed_numpy_3Darray.slided_numpy_array(s3r2)
            trainxs3_drill, trainys3_drill = windowed_numpy_3Darray.slided_numpy_array(s3_drill)
            valxs2r3, valys2r3 = windowed_numpy_3Darray.slided_numpy_array(s2r3)
            valxs3r3, valys3r3 = windowed_numpy_3Darray.slided_numpy_array(s3r3)
            testxs2r4, testys2r4 = windowed_numpy_3Darray.slided_numpy_array(s2r4)
            testxs2r5, testys2r5 = windowed_numpy_3Darray.slided_numpy_array(s2r5)
            testxs3r4, testys3r4 = windowed_numpy_3Darray.slided_numpy_array(s3r4)
            testxs3r5, testys3r5 = windowed_numpy_3Darray.slided_numpy_array(s3r5)

            trainx = np.concatenate(
                (trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
                 trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)

            trainy = np.concatenate(
                (trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
                 trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)

            valx = np.concatenate((valxs2r3, valxs3r3), axis=0)

            valy = np.concatenate((valys2r3, valys3r3), axis=0)

            testx = np.concatenate((testxs2r4, testxs2r5, testxs3r4, testxs3r5), axis=0)
            testy = np.concatenate((testys2r4, testys2r5, testys3r4, testys3r5), axis=0)

    del s1r1, s1r2, s1r3, s1r4, s1r5, s1_drill, s2r1, s2r2, s2r3, s2r4, s2r5, s2_drill, s3r1, s3r2, s3r3, s3r4, s3r5, s3_drill, s4r1, s4r2, s4r3, s4r4, s4r5, s4_drill
    return trainx, trainy, valx, valy, testx, testy