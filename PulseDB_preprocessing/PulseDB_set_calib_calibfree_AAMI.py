"""
File: PulseDB_DataFrame_Set.py
Project: 22HLT01 QUMPHY
Contact: mohammad.moulaeifard@uol.de
Gitlab: https://gitlab.com/qumphy
Description: Setting calb, calibfree and AAMI sets to the data frame.
SPDX-License-Identifier: EUPL-1.2
"""

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os

# location of the metadata.csv file
base_directory = "path/to/your/directory"

# Path for the metadata DataFrame file
metadata_file_path = os.path.join(base_directory, "metadata.csv")

# Load the merged DataFrame directly from metadata.csv
df_merged = pd.read_csv(metadata_file_path)

# Drop the Unnamed: 0 column if it exists
if 'Unnamed: 0' in df_merged.columns:
    df_merged = df_merged.drop(columns=['Unnamed: 0'])

# now we have a merged df file.    
df = df_merged    
##############################################################################################################################
# Configuration for pulsedb_calibfree
num_pat_calibfree_vital = 144
num_pat_calibfree_mimic = 135
num_pat_calibfree_vital_calib = 48 # 48 patients from vital for calib
num_pat_calibfree_mimic_calib = 45 # 45 patients from MIMIC for calib

# Creating validation and calibration set for pulsedb_calibfree
pat_train_vital = df[(df.set == 0) & (df.source == 0)].subject.unique()
pat_train_mimic = df[(df.set == 0) & (df.source == 1)].subject.unique()

pat_valcalib_calibfree_vital = np.random.permutation(pat_train_vital)[:num_pat_calibfree_vital]
pat_valcalib_calibfree_mimic = np.random.permutation(pat_train_mimic)[:num_pat_calibfree_mimic]

pat_calibfree_val = np.concatenate(
    (pat_valcalib_calibfree_vital[num_pat_calibfree_vital_calib:],
     pat_valcalib_calibfree_mimic[num_pat_calibfree_mimic_calib:])
)
pat_calibfree_calib = np.concatenate(
    (pat_valcalib_calibfree_vital[:num_pat_calibfree_vital_calib],
     pat_valcalib_calibfree_mimic[:num_pat_calibfree_mimic_calib])
)

# Assigning set_calibfree flags
df["set_calibfree"] = -1
df.loc[(df.set == 0) & df.subject.apply(lambda x: x not in np.concatenate((pat_calibfree_val, pat_calibfree_calib))), "set_calibfree"] = 0 # train
df.loc[(df.set == 0) & df.subject.apply(lambda x: x in pat_calibfree_val), "set_calibfree"] = 1 # val
df.loc[(df.set == 0) & df.subject.apply(lambda x: x in pat_calibfree_calib), "set_calibfree"] = 2 # calib
df.loc[df.set == 2, "set_calibfree"] = 3 # test

##############################################################################################################################
# Configuration for pulsedb_calib
num_samp_calib = 40
num_samp_calib_calib = 5 # use five segments for calib

pat_train = df[df.set == 0].subject.unique()
df["id"] = range(len(df))
id_list_val = []
id_list_calib = []

# Generating id_list for pulsedb_calib
for p in tqdm(pat_train):
    tmp = np.random.permutation(df[df.subject == p].id)[:num_samp_calib]
    id_list_val.append(tmp[num_samp_calib_calib:])
    id_list_calib.append(tmp[:num_samp_calib_calib])

id_list_val = np.concatenate(id_list_val)
id_list_calib = np.concatenate(id_list_calib)

# Assigning set_calib flags
df["set_calib"] = -1
df.loc[(df.set == 0) & df.id.apply(lambda x: x not in np.concatenate((id_list_val, id_list_calib))), "set_calib"] = 0 # train
df.loc[(df.set == 0) & df.id.apply(lambda x: x in id_list_val), "set_calib"] = 1 # val
df.loc[(df.set == 0) & df.id.apply(lambda x: x in id_list_calib), "set_calib"] = 2 # calib
df.loc[df.set == 1, "set_calib"] = 3 # test

# Cleaning up the DataFrame
df = df.drop(["id"], axis=1)

##############################################################################################################################
# Configuration for pulsedb_aami
num_pat_aami_vital_calib = 48 # 48 patients from vital for calib
num_pat_aami_mimic_calib = 45 # 45 patients from MIMIC for calib

# split val into val+calib randomly based on patients
pat_val_vital = np.random.permutation(df[(df.set == 3) & (df.source == 0)].subject.unique())
pat_val_mimic = np.random.permutation(df[(df.set == 3) & (df.source == 1)].subject.unique())

pat_aami_val = np.concatenate(
    (pat_val_vital[num_pat_aami_vital_calib:],
     pat_val_mimic[num_pat_aami_mimic_calib:])
)
pat_aami_calib = np.concatenate(
    (pat_val_vital[:num_pat_aami_vital_calib],
     pat_val_mimic[:num_pat_aami_mimic_calib])
)

# Assigning set_aami flags
df["set_aami"] = -1
df.loc[df.set == 0, "set_aami"] = 0 # train
df.loc[(df.set == 3) & df.subject.apply(lambda x: x in pat_aami_val), "set_aami"] = 1 # val
df.loc[(df.set == 3) & df.subject.apply(lambda x: x in pat_aami_calib), "set_aami"] = 2 # calib
df.loc[df.set == 4, "set_aami"] = 3 # test

##############################################################################################################################
# Adding the 'label' column
df["label"] = df["dbp_avg"].astype(str) + "," + df["sbp_avg"].astype(str)

##############################################################################################################################
# Saving the final DataFrame to the base directory
df.to_pickle(f"{base_directory}/df_memmap.pkl") # we just name it df_memmap for using it inside the main-ppg code. But you can put any name
df.to_csv(f"{base_directory}/metadata.csv", index=False) # Set index=False to avoid writing row names (indexes)
