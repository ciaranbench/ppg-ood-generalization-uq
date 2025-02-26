# Instructions for Preprocess_External_dataset.py Script

## Overview

This script is part of the **22HLT01 QUMPHY** project and is designed to preprocess scientific datasets, converting CSV and MAT files into NumPy and DataFrame formats.

## Prerequisites

- Ensure Python is installed on your system.
- Install the necessary libraries:
  ```sh
  pip install numpy pandas scipy
  ```
- Set up your dataset directories correctly before running the script.

## Script Functions

### 1. `process_folder(path: str, output_path: str) -> tuple`

This function processes a dataset folder by extracting features and signals.

- **Input**:
  - `path`: Path to the dataset folder.
  - `output_path`: Path where processed files will be saved.
- **Output**:
  - A DataFrame containing the processed data.
  - Lists of saved signal file paths.

### 2. `if __name__ == "__main__"`

Executes the script, processing multiple datasets and saving the combined data.

## Execution Guide

1. **Modify Dataset Paths** Edit the `input_dirs` dictionary inside the script to point to the correct dataset locations.

2. **Run the Script** Execute the script using:

   ```sh
   python Preprocess_SciData.py
   ```

3. **Output Files**

   - Processed dataset is saved as `df_mapped.pkl` in the specified output directory.
   - Signal files are saved as `.npy` files in the output folder.

## Notes

- Ensure your dataset folders contain correctly formatted CSV and MAT files.
- If you encounter missing files or format issues, check the dataset structure before running the script.
- The script automatically handles missing signal files and skips them without error.
