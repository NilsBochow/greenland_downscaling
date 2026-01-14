import xarray as xr
import numpy as np
from dataclasses import replace
from src.configuration import parse_command_line
from src.utils.transforms import fit_asinh_stats_xr, asinh_transform_xr  # make sure this is imported

# 1) Load config & training reference (ST/SMB) for TRAIN years only
config = parse_command_line()
modified_config = replace(config, transforms=['asinh'])  # kept for provenance only

da = xr.open_dataset(
    '/p/projects/ou/labs/ai/Nils/arctic-downscaling/data/datasets/CMIP6/SMB_ST_5km_MAR_merged_masked.nc'
)['ST'].load()

train_da = da.sel(time=da.time.dt.year.isin(config.train_years))

# 2) Robust asinh stats per pixel
#    s0: robust scale in data space; mu_z, sig_z: center/scale in z=asinh(y/s0) space
s0_da, mu_z_da, sig_z_da = fit_asinh_stats_xr(train_da)
z_train = asinh_transform_xr(train_da, s0_da)
mean_z  = z_train.mean('time')
std_z   = z_train.std('time')

# (Optional) also store simple per-pixel moments & ranges in data space
mu_y  = train_da.mean(dim='time')
std_y = train_da.std(dim='time')
lo_y  = train_da.min(dim='time')
hi_y  = train_da.max(dim='time')
print("data space stats:")
print(np.nanmin(mean_z.values), np.nanmax(mean_z.values), np.nanmean(mean_z.values))
print(np.nanmin(std_z.values), np.nanmax(std_z.values), np.nanmean(std_z.values))
print(np.nanmin(s0_da.values), np.nanmax(s0_da.values), np.nanmean(s0_da.values))

# 3) Package to a Dataset with provenance
ds_stats = xr.Dataset(
    {
        's0':    s0_da.astype('float32'),
        'mu_z':  mu_z_da.astype('float32'),
        'sig_z': sig_z_da.astype('float32'),
        'mean_y': mean_z.astype('float32'),
        'std_y':  std_z.astype('float32'),
        'lo_y':   lo_y.astype('float32'),
        'hi_y':   hi_y.astype('float32'),
    }
)

ds_stats.attrs.update({
    'transform_pipeline': 'asinh -> standardize(z)',
    'z_definition': 'z = asinh(y / s0); y in SMB units',
    'standardize_stats_space': 'z-space (mu_z, sig_z)',
    'train_years': ','.join(str(int(y)) for y in np.unique(train_da['time'].dt.year.values)),
})

# 4) Save once
out_path = '/p/projects/ou/labs/ai/Nils/arctic-downscaling/data/datasets/CMIP6/transform_stats_train_MAR_ST_asinh.nc'
ds_stats.to_netcdf(out_path)
print(f"wrote {out_path}")
