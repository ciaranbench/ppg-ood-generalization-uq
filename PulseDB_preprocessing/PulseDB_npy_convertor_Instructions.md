Instructions for PulseDB_npy_convertor.py Script
Overview
This script is part of the 22HLT01 QUMPHY project and is intended to handle PulseDB subset files, transforming them into NumPy and DataFrame files for further analysis.
Pre-requisites
- Ensure you have Python installed on your system.
- The script requires numpy, pandas, os, pathlib, and mat73 libraries. Make sure these are installed in your Python environment.
Script Functions
1. determine_set_value(file_name: str) -> int
This function categorizes the input file as training or testing data based on its name.
- Input: The file name (a string).
- Output: An integer representing the data category (0 for training, 1-4 for different types of testing, -1 if the category is undetermined).
2. build_dataset(input_path: str, save_path: str, field_name='Subset') -> str
This function converts data from a .mat file into .npy files for each data component within the specified field.
- Input:
  - input_path: Path to the .mat file.
  - save_path: Directory path where the .npy files will be saved.
  - field_name: Key name in the .mat file to access the data (defaults to 'Subset').
- Output: The path where the .npy files are saved.
3. process_npy_files(path: str, set_value: int) -> pd.DataFrame
This function compiles .npy files from a directory into a single pandas DataFrame.
- Input:
  - path: Directory containing the .npy files.
  - set_value: Set value that categorizes the data as training or testing.
- Output: A pandas DataFrame with all the compiled data.
Execution Guide
1. Set File Paths: Modify input_mat_path and output_npy_path in the script's main section to point to your .mat file and desired output directory, respectively.
2. Running the Script: Execute the script using a Python interpreter. It performs the following actions:
   - Calls build_dataset to convert the .mat file to .npy files and save them to the specified directory.
   - Determines the set value (training or testing) using determine_set_value.
   - Processes the .npy files into a DataFrame with process_npy_files.
   - Saves the DataFrame in CSV and Pickle formats in the specified output directory.
3. Output: The script outputs several .npy files, one for each data component in the input .mat file, and a DataFrame in both CSV and Pickle formats containing the compiled data.
Notes
- Modify the input_mat_path and output_npy_path variables in the script's if __name__ == "__main__" block to match your file locations.
- Ensure the .mat file structure aligns with the script's expectations (i.e., it contains the specified field_name and data structure).
