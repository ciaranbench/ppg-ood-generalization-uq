## PulseDB & External dataset

### Initial Setup

1. **Download Required Files**:
   - For each dataset, first generate the corresponding `signals.npy` and `metadata.csv` or `metadata.pkl` files using preprocessing instructions.

2. **Organize Files**:
   - Create a folder named `data` in your project directory.
   - Place the downloaded `signals.npy` and `metadata.csv` files inside this `data` folder.

### Memmap and df_memmap Files Generation

1. **Run the Conversion Script**:
   - Use the `memmap_df_conversion.py` script to generate the `memmap` and `df_memmap` files.

2. **Generated Files**:
   - After running the script, the following files will be generated in the `data` folder:
     - `df_memmap.pkl`
     - `memmap.npy`
     - `memmap_meta.npz`

### Final File Organization

1. **Create a New Folder**:
   - Create a new folder to store the final set of files (this is a folder that you ahould address it later in the main-ppg code) 

2. **Move Generated Files**:
   - Move the generated `df_memmap.pkl`, `memmap.npy`, and `memmap_meta.npz` files to this new folder.

3. **Download Additional Files**:
   - From the repository, download the following files located in the "extra required files" folder:
     - `mean.npy`
     - `lbl_itos.npy`
     - `std.npy`

4. **Complete the Final Folder**:
   - Place the `mean.npy`, `lbl_itos.npy`, and `std.npy` files into the new folder created in step 1.
   - This folder should now contain the following six files:
     - `df_memmap.pkl`
     - `memmap.npy`
     - `memmap_meta.npz`
     - `mean.npy`
     - `lbl_itos.npy`
     - `std.npy`

5. **Specify Path for Use**:
   - When you need to access the dataset, specify the path to this final folder using the `--data` flag in the terminal.

# Examples for processing the datasets 


# Training models BP (Calib Vital)


| key        | value                                                                                                                                                                       |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset    | PulseDB                                                                                                                                                                     |
| model      | BaseLine/XResNet1d101/XResNet1d50/Inception1d/LeNet1d                                                                                                                       |
| script     | .../required_codes_files/main_ppg.py                     |
|train command    | `python main_ppg.py --data ./path/to/folder/with/six/final/files --input-size 1250 --architecture XResNet1d101/XResNet1d50/Inception1d --finetune-dataset pulsedb_calib_vital  --select-input-channel 0 --refresh-rate 1 --batch-size 512 --epoc 50`  |
| comment    |        |

# Training models BP (CalibFree Vital)

| key        | value                                                                                                                                                                           |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset    | PulseDB                                                                                                                                                                         |
| model      | BaseLine/XResNet1d101/XResNet1d50/Inception1d/LeNet1d                                                                                                                           |
| script     | .../required_codes_files/main_ppg.py                           |
| train command    | `python main_ppg.py --data ./path/to/folder/with/six/final/files --input-size 1250 --architecture XResNet1d101/XResNet1d50/Inception1d --finetune-dataset pulsedb_calibFree_vital  --select-input-channel 0 --refresh-rate 1 --batch-size 512 --epoc 50`  |
| comment    |               |


# Training models BP (BCG/UCI/Sensors/PPGBP)

| key        | value                                                                                                                                                                           |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset    | External dataset                                                                                                                                                                         |
| model      | BaseLine/XResNet1d101/XResNet1d50/Inception1d/LeNet1d                                                                                                                           |
| script     | .../required_codes_files/main_ppg.py                           |
| train command    | `python main_ppg.py --data ./path/to/folder/with/six/final/files --input-size 625/625/625/256 --architecture XResNet1d101/XResNet1d50/Inception1d --finetune-dataset bpbenchmark_bcg/bpbenchmark_uci/bpbenchmark_sensors/bpbenchmark_ppgbp  --select-input-channel 0 --refresh-rate 1 --batch-size 512 --epoc 50`  |
| comment    |               |


# Training weighted models BP (train on AAMI_Vital , test on Calib_MIMIC)


| key        | value                                                                                                                                                                       |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset    | PulseDB                                                                                                                                                                     |
| model      | BaseLine/XResNet1d101/XResNet1d50/Inception1d/LeNet1d                                                                                                                       |
| script     | .../required_codes_files/main_ppg_weighted.py                     |
|train command    | `python main_ppg_weighted.py --data ./path/to/folder/with/six/final/files --input-size 1250 --architecture XResNet1d101/XResNet1d50/Inception1d --finetune-dataset pulsedb_aami_vital  --select-input-channel 0 --refresh-rate 1 --batch-size 512 --epoc 50  --sbp-weights-file $WEIGHTS_PATH/"weights_train_SBP_aami_vital_test_on_calib_mimic_test" --bins-file $BINS_PATH/"bins.npy" --dbp-weights-file $WEIGHTS_PATH/"weights_train_DBP_aami_vital_test_on_calib_mimic_test.npy"`|
| comment    |        |

