"""
File: PulseDB_PPG_signal_metadata_merging.py
Project: 22HLT01 QUMPHY
Contact: mohammad.moulaeifard@uol.de
Gitlab: https://gitlab.com/qumphy
Description: Merging PulseDB ppg signals and df files
SPDX-License-Identifier: EUPL-1.2
"""

import os
import pandas as pd
import numpy as np

def merge_dataframes(directory_path, train_csv, test_csvs):
    """
    Merge DataFrames from training and test CSV files.

    Parameters
    ----------
    directory_path : str
        The directory containing the CSV files.
    train_csv : str
        The filename of the training DataFrame CSV file.
    test_csvs : list of str
        List of filenames for the test DataFrame CSV files.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame.
    """
    # Load the training DataFrame
    train_df = pd.read_csv(os.path.join(directory_path, train_csv))
    
    # Sequentially merge test DataFrames with the training DataFrame
    for test_csv in test_csvs:
        test_df = pd.read_csv(os.path.join(directory_path, test_csv))
        train_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Save the merged DataFrame to CSV format
    merged_csv = os.path.join(directory_path, 'metadata.csv')
    train_df.to_csv(merged_csv, index=False)
    
    return merged_csv

def merge_signals(directory_path, signal_files):
    """
    Merge memory-mapped numpy files.

    Parameters
    ----------
    directory_path : str
        The directory containing the numpy files.
    signal_files : list of str
        List of filenames for the memory-mapped numpy files.

    Returns
    -------
    str
        The filename of the merged memory-mapped numpy file.
    """
    # Load signals from each .npy file in the specified order
    signals_list = [np.load(os.path.join(directory_path, file)) for file in signal_files]
    
    # Concatenate signals along axis 0 (assuming they have the same shape)
    merged_signals = np.concatenate(signals_list, axis=0)
    
    # Save the merged signals to a new .npy file
    merged_npy = os.path.join(directory_path, 'signals.npy')
    np.save(merged_npy, merged_signals)
    
    return merged_npy

def main():
    directory_path = './path/to/files/'
    train_csv = 'df_train.csv'
    test_csvs = ['df_test_1.csv', 'df_test_2.csv', 'df_test_3.csv', 'df_test_4.csv']
    
    signal_files = [
        'signals_f_ppg_train.npy',
        'signals_f_ppg_test_1.npy',
        'signals_f_ppg_test_2.npy',
        'signals_f_ppg_test_3.npy',
        'signals_f_ppg_test_4.npy'
    ]

    # Merge DataFrames
    merged_csv = merge_dataframes(directory_path, train_csv, test_csvs)
    print(f"Merged DataFrame saved to: {merged_csv}")

    # Merge signals
    merged_npy = merge_signals(directory_path, signal_files)
    print(f"Merged signals saved to: {merged_npy}")

if __name__ == "__main__":
    main()
