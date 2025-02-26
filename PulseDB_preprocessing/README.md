
# Preprocessing the PulseDB (Vital and MIMIC Datasets): A Step-by-Step Guide


## 1. Downloading the Initial Data

Begin by downloading the raw data and creating the associated subset files from the main [PulseDB GitHub page](https://github.com/pulselabteam/PulseDB?tab=readme-ov-file). Follow the instructions provided on the page to download all the required data. After downloading, you can either proceed with the original instructions involving MATLAB and Python codes to generate the .npy files or implement a few adjustments to create lighter files tailored to the QUMPHY project's objectives. The adjustments will be outlined in the following sections.



## 2. Generating the Subset Files from Initial Data

Once you've downloaded the necessary files, the GitHub instructions suggest running `Generate_Subsets.m` to produce subset files. We have slightly modified this script to fit our framework. Please use `Generate_Subsets_QUMPHY.m` instead, which will generate five new .m files:

- `Train_Subset.m`
- `CalBased_Test_Subset.m`
- `CalFree_Test_Subset.m`
- `AAMI_Test_Subset.m`
- `AAMI_Cal_Subset.m`



## 3. Generating the .npy and DataFrame Files for Each Subset File

Next, you need to generate the .npy and .df files based on the five .m files generated in the previous step. To generate these files, execute `PulseDB_npy_convertor.py` and address each of the five .m files in the `input_mat_path` of this code. This procedure will generate 15 new .npy files for each dataset, containing information on the signal, DBP, SBP, gender, age, BMI, caseID, source (MIMIC or Vital), weight, etc. Additionally, you will have one .csv/.pkl file as the specific metadata for each set.



## 4. Generation of Merged PPG Signals and Metadata Files

To merge the PPG signals and .df (metadata) files into two comprehensive files, follow these steps:

1. Create an empty folder.
2. Copy the following 10 files into this empty folder:
   - `signals_f_ppg_train.npy`
   - `signals_f_ppg_test_1.npy`
   - `signals_f_ppg_test_2.npy`
   - `signals_f_ppg_test_3.npy`
   - `signals_f_ppg_test_4.npy`
   - `df_train.csv`
   - `df_test_1.csv`
   - `df_test_2.csv`
   - `df_test_3.csv`
   - `df_test_4.csv`

Next, run `PulseDB_PPG_signal_metadata_merging.py` by providing the address of the folder you created. This will generate the merged files `signals.npy` and `metadata.csv`.



## 5. Defining the Calibration (Calib), Calibration-Free (Calibfree), and AAMI Sets

To define the Calib, Calibfree, and AAMI sets within the merged `metadata.csv` file, use `PulseDB_set_calib_calibfree_AAMI.py`. This code will add the Calib, Calibfree, and AAMI sets to the `metadata.csv` file. Import the address of the folder you generated in the previous step into this code. The code will find the `metadata.csv` file in this folder and update it accordingly.
