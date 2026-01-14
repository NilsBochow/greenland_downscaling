import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
factor=16
# 1. open
file = f'/data/datasets/test_set_ts_MAR_adaptive_32x_nanmean.nc'

def fill_2d_nearest(arr2d):
    """
    Fill NaNs in a 2‑D numpy array arr2d by nearest non‑NaN neighbor.
    """
    valid_mask = ~np.isnan(arr2d)
    if valid_mask.all():
        return arr2d  # nothing to fill
    # compute, for every pixel, the indices of the nearest valid pixel
    _, (i_inds, j_inds) = distance_transform_edt(
        ~valid_mask, return_indices=True
    )
    filled = arr2d.copy()
    nan_mask = ~valid_mask
    filled[nan_mask] = arr2d[i_inds[nan_mask], j_inds[nan_mask]]
    return filled

# 1) load & linear‑interp as before
ds = xr.open_dataset(file)
new_x = np.linspace(ds.x.min(), ds.x.max(), 336)
new_y = np.linspace(ds.y.min(), ds.y.max(), 576)
ds_interp = ds.interp(x=new_x, y=new_y, method="linear")

# 2) mask out your “bad” >1e8 values
ds_interp = ds_interp.where(ds_interp <= 1e8)

# 3) allocate a dict for the filled variables
filled = {}

for var in ds_interp.data_vars:
    da = ds_interp[var]
    data = da.values         # shape e.g. (time, y, x)
    out  = data.copy()
    # loop over time (or any leading dims)
    for idx in range(data.shape[0]):
        out[idx, :, :] = fill_2d_nearest(data[idx, :, :])
    # wrap back into a DataArray
    filled[var] = xr.DataArray(
        out, dims=da.dims, coords=da.coords, name=var
    )

# 4) build the final Dataset & save
ds_filled = xr.Dataset(filled, coords=ds_interp.coords)

# now save
output = f'/data/datasets/CMIP6/smb_ts_coarse_test_set/test_set_ts_MAR_adaptive_32x_linear.nc'

ds_filled.to_netcdf(output)

