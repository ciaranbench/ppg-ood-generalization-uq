
Instructions for Using memmap_conversion.py
===========================================

1. Ensure Dependencies:
   Make sure you have the required dependencies installed. You need `numpy`, `pandas`, and `timeseries_utils`. 
   You can install them using pip:
   
   ```bash
   pip install numpy pandas
   ```

2. Organize Files:
   Please put the signals.npy and metadata.csv into data folder and now ensure that your project directory has the following structure:
   
   your_project_directory/
   ├── memmap_conversion.py
   └── data/
       ├── signals.npy
       └── metadata.csv

3. Understand the Input Data:
   - `signals.npy`: This file contains the signal data in numpy array format. It is expected to be a large file containing time-series data that you want to convert into a memory-mapped file for efficient processing.
   - `metadata.csv`: This file contains metadata information related to the signal data. It is expected to be in CSV format and will be converted into a pandas DataFrame and then pickled.

4. What the Script Does:
   - The script reads the `signals.npy` file and converts it into a memory-mapped file named `memmap.npy` using the `npys_to_memmap_batched` function from the `timeseries_utils` module. This conversion allows for efficient access to large datasets by loading only necessary parts into memory.
   - The script then reads the `metadata.csv` file into a pandas DataFrame.
   - The DataFrame is saved as a pickle file named `df_memmap.pkl` for efficient storage and retrieval.

5. Run the Script:
   Navigate to the directory containing `memmap_conversion.py` and run the script:
   
   ```bash
   python memmap_conversion.py
   ```

The script will generate `memmap.npy` and `df_memmap.pkl` in the `data` directory.
