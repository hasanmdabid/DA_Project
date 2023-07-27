'''
General information
----------

This script provides global variables for the analysis of the ``Painmonit'' and ``Biovid'' datasets.
'''

#######################################################################################################################
# Variables
#######################################################################################################################

# Sampling rate of the PainMonit dataset
sampling_rate_painmonit = 250
# Sampling rate of the Biovid dataset
sampling_rate_biovid = 512
# Amount of seconds per stimulus
window_secs = 10
# Baseline temperature of the Medoc
baseline_temp= 32
# Number of repetitions per stimuli - uzl
num_repetitions_uzl = 8
# Name of the used sensors in PainMonit - these are the headers in for the np files
painmonit_sensors = ["Bvp", "Eda_E4", "Resp", "Eda_RB", "Ecg", "Emg"]
# Dict to translate a sensor names to it's index
painmonit_sensor_to_index = {j:i for i, j in enumerate(painmonit_sensors)}
# Name of all biovid sensors
biovid_sensors = ["time", "gsr", "ecg", "emg_trapezius", "emg_corrugator", "emp_zygomaticus"]
# Subjects to skip in UzL data
uzl_faulty = ["S_01", "S_22", "S_23", "S_24", "S_25", "S_33", "S_58"]
# Biosignals directory for the BioVid dataset - can be 'biosignals_filtered' or 'biosignals_raw'
biosignals_dir = "biosignals_raw"

num_painmonit_subjects = 52
num_biovid_subjects = 87