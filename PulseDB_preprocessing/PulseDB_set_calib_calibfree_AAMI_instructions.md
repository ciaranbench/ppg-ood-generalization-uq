# PulseDB DataFrame and Signal Processing

## Overview

This script processes a merged DataFrame (`metadata.csv`) and sets calibration, calibfree, and AAMI sets to the data frame. It also adds a new column `label` that concatenates the values of `dbp_avg` and `sbp_avg` columns.

## Prerequisites

- Python 3.x
- Pandas
- Numpy
- tqdm

## Setup

1. Ensure you have the following files in a single directory:
   - `metadata.csv` (this should be your merged DataFrame)

2. Replace `"path/to/your/directory"` in the script with the actual path to the directory containing your `metadata.csv` file.

