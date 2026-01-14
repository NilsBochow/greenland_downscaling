import xarray as xr
from src.configuration import parse_command_line
import numpy as np 
from src.configuration import Config
from dataclasses import replace
from src.utils.transforms import apply_transforms
# load once
config = parse_command_line()
modified_config = replace(config, transforms=['log'])
data_ref = xr.open_dataset('/p/projects/ou/labs/ai/Nils/arctic-downscaling/data/datasets/CMIP6/SMB_ST_5km_MAR_merged_masked.nc')["ST"][:] 
data_ref = data_ref.sel(time=data_ref.time.dt.year.isin(config.train_years)) 
epsilon      = config.epsilon
data_ref_min = data_ref.min()
#offset       = max(epsilon, epsilon - data_ref_min)

# per‐cell statistics
μ   = data_ref.mean(dim='time')
σ   = data_ref.std(dim='time')
lo  = data_ref.min(dim='time')
hi  = data_ref.max(dim='time')
log_transf_data_ref = apply_transforms(data = data_ref, data_ref = data_ref, config = modified_config)
log_mean = log_transf_data_ref.mean(dim='time')
log_std = log_transf_data_ref.std(dim='time')

# pack into a tiny Dataset
ds_stats = xr.Dataset({
    'data_ref_min': xr.DataArray(data_ref_min),
    'mean':      μ,
    'std':      σ,
    'lo':     lo,
    'hi':     hi,
    'log_mean': log_mean,
    'log_std': log_std
})

# save once
ds_stats.to_netcdf('/p/projects/ou/labs/ai/Nils/arctic-downscaling/data/datasets/CMIP6/transform_stats_train_MAR_TS.nc')
