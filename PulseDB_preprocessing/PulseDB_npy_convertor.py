"""
File: PulseDB_npy_convertor.py
Project: 22HLT01 QUMPHY
Contact: mohammad.moulaeifard@uol.de
Gitlab: https://gitlab.com/qumphy
Description: Convert PulseDB subset files to npy and df files
SPDX-License-Identifier: EUPL-1.2
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from mat73 import loadmat

def determine_set_value(file_name: str) -> int:
    """
    Determine the set value based on the input file name. This value is used to categorize
    the file as training or testing data.

    Parameters
    ----------
    file_name : str
        The name of the input file.

    Returns
    -------
    int
        The determined set value corresponding to the file type.
    """
    mapping = {
        "Train_Subset": 0,
        "CalBased_Test_Subset": 1,
        "CalFree_Test_Subset": 2,
        "AAMI_Cal_Subset": 3,
        "AAMI_Test_Subset": 4
    }
    for key, value in mapping.items():
        if key in file_name:
            return value
    return -1  # Default value if none of the keys match

def build_dataset(input_path: str, save_path: str, field_name='Subset') -> str:
    """
    Converts the data from a .mat file into separate .npy files for each data component
    within the specified field of the .mat file.

    Parameters
    ----------
    input_path : str
        Path to the .mat file.
    save_path : str
        Directory path where the .npy files will be saved.
    field_name : str
        The key name in the .mat file to access the data.

    Returns
    -------
    str
        The path where the .npy files are saved.
    """
    assert os.path.splitext(input_path)[-1] == ".mat"
    save_path = os.path.abspath(save_path)
    os.makedirs(save_path, exist_ok=True)
    Data = loadmat(input_path)[field_name]

    set_value = determine_set_value(os.path.basename(input_path))
    file_suffix_mapping = {
        0: "_train",
        1: "_test_1",
        2: "_test_2",
        3: "_test_3",
        4: "_test_4"
    }
    file_suffix = file_suffix_mapping.get(set_value, "_unknown")

    for key in Data.keys():
        data_array = np.array(Data[key]).squeeze()
        if key == 'Gender':
            data_array = (data_array == 'M').astype(float)
        elif key == 'Signals_F_PPG':
            data_array = data_array.astype('float32')
        np.save(os.path.join(save_path, f"{key.lower()}{file_suffix}.npy"), data_array)

    return save_path

def process_npy_files(path: str, set_value: int) -> pd.DataFrame:
    """
    Processes .npy files in the given directory into a single pandas DataFrame.

    Parameters
    ----------
    path : str
        The directory containing the .npy files.
    set_value : int
        The set value that categorizes the data as training or testing.

    Returns
    -------
    pd.DataFrame
        The compiled DataFrame containing all the data.
    """
    path = Path(path)
    npy_files = sorted(list(path.glob('*.npy')))
    res = {}
    for file_name in npy_files:
        _, tail = os.path.split(file_name)
        field = "_".join(tail[:-4].split("_")[:-1]).replace("id", "_id").replace("dbp", "dbp_avg").replace("sbp", "sbp_avg")
        if "signals" not in field:
            res[field] = np.load(file_name).tolist()
            res["set"] = [set_value] * len(res[field])
    
    df = pd.DataFrame(res)
    new_columns = {col: col.rsplit('_test', 1)[0] for col in df.columns if '_test' in col}
    df.rename(columns=new_columns, inplace=True)
    
    df["data"] = np.arange(len(df))
    print(df)
    return df

if __name__ == "__main__":
    input_mat_path = 'path/to/input_file.mat'
    output_npy_path = 'path/to/save_directory'
    build_dataset(input_mat_path, output_npy_path)

    set_value = determine_set_value(os.path.basename(input_mat_path))
    base_output_name = "df_train" if set_value == 0 else f"df_test_{set_value}"
    
    csv_save_path = os.path.join(output_npy_path, f"{base_output_name}.csv")
    pickle_save_path = os.path.join(output_npy_path, f"{base_output_name}.pkl")
    
    df = process_npy_files(output_npy_path, set_value)
    df.to_csv(csv_save_path, index=False)
    df.to_pickle(pickle_save_path)
    print(f"DataFrame saved to: {csv_save_path} and {pickle_save_path}")