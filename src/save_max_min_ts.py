from typing import List, Optional, Type, Union, Tuple

import torch 
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from dataclasses import dataclass
from dataclasses import replace
from matplotlib import pyplot as plt
from src.utils.transforms import apply_transforms
import pandas as pd


np.random.seed(42)
years = np.arange(1950, 2101)  # Including 2100

# Shuffle the years
np.random.shuffle(years)
plot_valid_samples = True
# Define split ratios
#show_valid_samples_tensorboard = True
plot_valid_samples = True
train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.15

# Compute split indices
n_total = len(years)
n_train = int(n_total * train_ratio)
n_valid = int(n_total * valid_ratio)

# Assign years to sets
train_years = years[:n_train] 
valid_years = years[n_train:n_train + n_valid] 
test_years =  years[n_train + n_valid:] 


target = xr.open_dataset("/data/datasets/CMIP6/SMB_ST_5km_MAR_merged_masked.nc", use_cftime=True)["ST"]


target = target.sel(time=target.time.dt.year.isin(train_years))
min_val = target.min()
max_val = target.max()

# If these are DataArray objects with one element, you can extract the scalar value using .item()
min_scalar = min_val.item()
max_scalar = max_val.item()

# Create a DataFrame with the min and max values
df = pd.DataFrame({
    'min': [min_scalar],
    'max': [max_scalar]
})

# Save the DataFrame to a CSV file
csv_filename = "/data/TS_train_min_max_values.csv"
df.to_csv(csv_filename, index=False)

print(f"Saved minimum and maximum values to {csv_filename}")

