import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

# —— parameters —— 
file_path  = '/data/datasets/CMIP6/smb_ts_coarse_test_set/test_set_MAR.nc'
factor     = 32
base_out   = '/data/datasets/CMIP6/smb_ts_coarse_test_set/'

now_str    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# —— load & mask zeros as NaN —— 
ds = xr.open_dataset(file_path)
ds = ds.where(ds != 0, other=np.nan)

# —— compute trimmed dims & coarsened dims —— 
y_dim, x_dim = ds.sizes['y'], ds.sizes['x']
y_trim = (y_dim // factor) * factor
x_trim = (x_dim // factor) * factor
y_coarse = y_trim // factor
x_coarse = x_trim // factor

# —— precompute low‐res coords (block‐centroids) —— 
y_vals = (
    ds.coords['y'][:y_trim].values
    .reshape(y_coarse, factor)
    .mean(axis=1)
)
x_vals = (
    ds.coords['x'][:x_trim].values
    .reshape(x_coarse, factor)
    .mean(axis=1)
)
coords_low = {
    'time': ds.coords['time'],
    'y':      ('y', y_vals),
    'x':      ('x', x_vals),
}
coords_full = ds.coords  # original

lowres_vars = {}
upsamp_vars = {}

for var in ds.data_vars:
    # --- pull array and trim so dims divisible by factor ---
    arr = ds[var].values.astype('float32')         # (time, y, x)
    arr_t = arr[:, :y_trim, :x_trim]               # trim edges

    # --- build torch tensors & mask ---
    t     = torch.from_numpy(arr_t).unsqueeze(1)   # (time,1,y_trim,x_trim)
    mask  = (~torch.isnan(t)).float()              # 1 where valid
    t0    = torch.nan_to_num(t, 0.0)               # zero out NaNs

    # --- adaptive‐avg pooling: sum & count per block ---
    out_size = (y_coarse, x_coarse)
    # average of t0*mask times block‐area = sum
    sum_pool   = F.adaptive_avg_pool2d(t0*mask, out_size) * (factor*factor)
    # average of mask times block‐area = count of valid pixels
    count_pool = F.adaptive_avg_pool2d(mask, out_size) * (factor*factor)

    # --- compute nan‐mean, set fully‐masked blocks back to NaN ---
    mean_coarse = sum_pool / torch.clamp(count_pool, min=1.0)
    mean_coarse[count_pool == 0] = float('nan')

    # --- nearest‐neighbor upsample to original grid ---
    up = F.interpolate(mean_coarse, size=(y_dim, x_dim), mode='nearest')

    # --- back to numpy & drop channel dim ---
    arr_coarse = mean_coarse.squeeze(1).cpu().numpy()  # (time,y_coarse,x_coarse)
    arr_up     = up.squeeze(1).cpu().numpy()           # (time,y,x)

    # --- wrap as DataArray ---
    lowres_vars[f'{var}'] = xr.DataArray(
        arr_coarse,
        coords=coords_low,
        dims=('time','y','x'),
        attrs=ds[var].attrs
    )
    upsamp_vars[var] = xr.DataArray(
        arr_up,
        coords=coords_full,
        dims=('time','y','x'),
        attrs=ds[var].attrs
    )

# —— build Datasets & add history —— 
ds_lowres = xr.Dataset(lowres_vars, attrs=ds.attrs)
ds_up     = xr.Dataset(upsamp_vars, attrs=ds.attrs)
history = (
    f"{now_str}: adaptive avg‐pooled by factor {factor} (ignoring NaNs), "
    "then nearest‐neighbor upsample."
)
ds_lowres.attrs['history'] = history
ds_up    .attrs['history'] = history

# —— save to NetCDF —— 
low_path = f"{base_out}/test_set_ts_MAR_adaptive_{factor}x_nanmean.nc"
up_path  = f"{base_out}/test_set_ts_MAR_adaptive_{factor}x_upsampled.nc"

#ds_lowres.to_netcdf(low_path)
ds_up    .to_netcdf(up_path)
ds_lowres.to_netcdf(low_path)
print(f"Low‐res file saved to:    {low_path}")
print(f"Upsampled file saved to:  {up_path}")
