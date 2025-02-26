"""
File: memmap_conversion.py
Project: 22HLT01 QUMPHY
Contact: mohammad.moulaeifard@uol.de, nils.strodthoff@uol.de
Gitlab: https://gitlab.com/qumphy
Description: memmap conversion and df_memmap generation
SPDX-License-Identifier: EUPL-1.2
"""

import numpy as np
import pandas as pd
import os
from timeseries_utils import npys_to_memmap_batched

# Define the base directory and file paths
base_dir = os.path.join(os.getcwd(), 'data')
signals_path = os.path.join(base_dir, 'signals.npy')
memmap_path = os.path.join(base_dir, 'memmap.npy')
metadata_path = os.path.join(base_dir, 'metadata.csv')
pkl_path = os.path.join(base_dir, 'df_memmap.pkl')

# Use npys_to_memmap_batched to create a memory-mapped file
npys_to_memmap_batched(
    [signals_path],  # Pass as a list of paths
    memmap_path,
    max_len=0,
    delete_npys=False,
    batched_npy=True,
    batch_length=100000  # Adjust batch_length as needed for memory usage
)

# Load metadata.csv
metadata_df = pd.read_csv(metadata_path)

# Save the DataFrame as a pickle file
metadata_df.to_pickle(pkl_path)

print("Conversion completed successfully.")
