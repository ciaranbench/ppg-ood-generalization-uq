"""
File: weighting_calculation.py
Project: 22HLT01 QUMPHY
Contact: mohammad.moulaeifard@uol.de
Gitlab: https://gitlab.com/qumphy
Description: Compute weights & similarity measures between SBP and DBP distributions from different datasets.
SPDX-License-Identifier: EUPL-1.2
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance



# Define the source and target dataset names
source_names = [
    "aami_combined_test", "calibfree_combined_test", "calib_combined_test",
    "calibfree_vital_test", "calib_vital_test", "calib_mimic_test",
    "calibfree_mimic_test", "aami_vital_test", "aami_mimic_test"
]

target_names = [
    "aami_combined_test", "calibfree_combined_test", "calib_combined_test",
    "sensors_dataset", "uci2_dataset", "ppgbp_dataset", "bcg_dataset",
    "calibfree_vital_test", "calib_vital_test", "calib_mimic_test",
    "calibfree_mimic_test", "aami_vital_test", "aami_mimic_test"
]

# Define the number of bins and the fixed bin range (0 to 280)
num_bins = 200
fixed_bins = np.linspace(0, 280, num_bins + 1)
np.save("bins.npy", fixed_bins)
print("Fixed bins saved to bins.npy")

# Lists to collect results for SBP and DBP
results_sbp = []
results_dbp = []

def process_blood_pressure(bp_main, bp_external, bp_type, source_name, target_name):
    hist_main, _ = np.histogram(bp_main, bins=fixed_bins, density=True)
    hist_external, _ = np.histogram(bp_external, bins=fixed_bins, density=True)

    weights = np.zeros(len(hist_main))
    min_val = 1
    alpha = 100.0

    for i in range(len(hist_main)):
        if hist_main[i] > 0:
            weights[i] = max(min_val, alpha * hist_external[i] / hist_main[i])
        else:
            weights[i] = min_val

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    emd_similarity = wasserstein_distance(hist_main, 100 * hist_external)
    
    output_filename = f"weights_train_{bp_type}_{source_name}_on_{target_name}.npy"
    np.save(output_filename, weights_tensor.numpy())
    print(f"Weights saved to {output_filename}")
    
    return {
        'Source': source_name,
        'Target': target_name,
        'Earth Mover\'s Distance (EMD)': emd_similarity
    }

for source_name in source_names:
    for target_name in target_names:
        print(f"Processing: Source={source_name}, Target={target_name}")

        source = pd.read_pickle(f"{source_name}.pkl")
        target = pd.read_pickle(f"{target_name}.pkl")

        sbp_main = source['sbp_avg'].values
        dbp_main = source['dbp_avg'].values

        if target_name in ["sensors_dataset", "uci2_dataset", "ppgbp_dataset", "bcg_dataset"]:
            sbp_external = target['sp'].values
            dbp_external = target['dp'].values
        else:
            sbp_external = target['sbp_avg'].values
            dbp_external = target['dbp_avg'].values

        results_sbp.append(process_blood_pressure(sbp_main, sbp_external, 'SBP', source_name, target_name))
        results_dbp.append(process_blood_pressure(dbp_main, dbp_external, 'DBP', source_name, target_name))

print("All similarity results have been computed.")


# Exploiting the weights

you can easily now use the weights in traning the models using "main_ppg_weighted". For example 


# Training weighted models BP (train on AAMI_Vital , test on Calib_MIMIC)


| key        | value                                                                                                                                                                       |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset    | PulseDB                                                                                                                                                                     |
| model      | BaseLine/XResNet1d101/XResNet1d50/Inception1d/LeNet1d                                                                                                                       |
| script     | .../required_codes_files/main_ppg_weighted.py                     |
|train command    | `python main_ppg_weighted.py --data ./path/to/folder/with/six/final/files --input-size 1250 --architecture XResNet1d101/XResNet1d50/Inception1d --finetune-dataset pulsedb_aami_vital  --select-input-channel 0 --refresh-rate 1 --batch-size 512 --epoc 50  --sbp-weights-file $WEIGHTS_PATH/"weights_train_SBP_aami_vital_test_on_calib_mimic_test" --bins-file $BINS_PATH/"bins.npy" --dbp-weights-file $WEIGHTS_PATH/"weights_train_DBP_aami_vital_test_on_calib_mimic_test.npy"`|
| comment    |        |

