import xarray as xr
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Mask every variable in an SMB NetCDF file.')
parser.add_argument('--smb_file',   type=str, required=True,
                    help='Path to the SMB NetCDF file')
parser.add_argument('--model_name', type=str, required=True,
                    help='Model Name (used in output filename)')
args = parser.parse_args()

# 1) Open your main dataset
ds = xr.open_dataset(args.smb_file)

# 2) Select mask file exactly as before
#    (here res is hardcoded to "high" to match your original logic)
res = "high"
if res == "lowres":
    mask_file = "data/datasets/mask_sftgif.nc"
    mask_var  = "sftgif"
else:
    mask_file = "data/datasets/mask_file_1.nc"
    mask_var  = "MSK"

ds_mask = xr.open_dataset(mask_file, decode_times=False)
msk     = ds_mask[mask_var]

# 3) Build a boolean array: True where we want to keep the data
keep = (msk.values >= 4)

# 4) Loop over *all* data variables and zero‐out where mask is False
threshold = 1_000_000
print(ds.data_vars)
for var in ["ts"]:
    da = ds[var]
    # only apply mask to variables whose dims cover the mask dims
    if set(msk.dims).issubset(da.dims):
        # first mask out unwanted grid cells (keep is your boolean mask)
        da = da.where(keep, 0)
        # then zero out any values whose magnitude exceeds threshold
        da = da.where(np.abs(da) <= threshold, 0)
        # write back into the dataset
        ds[var] = da
# 5) Write out
output_file = f"{args.model_name}_masked.nc"
ds.to_netcdf(output_file)
ds.close()

print(f"All variables masked and saved to {output_file}")
