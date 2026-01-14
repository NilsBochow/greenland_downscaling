#!/usr/bin/env python3
"""
compute_spatial_crps_from_files.py

Compute grid-point CRPS when your ensemble members are each stored
in separate NetCDF files, each with dims (time, y, x).

This version:
  - Lazy‐loads each file with Dask chunks
  - Builds a true (member, time, y, x) array
  - Ensures “member” is a core dim
  - Calls xskillscore without ever OOMing
"""

import argparse
import glob

import xarray as xr
import xskillscore as xs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs",       required=True,
                   help="Observation NetCDF file (dims: time,y,x)")
    p.add_argument("--fc-pattern",required=True,
                   help="Glob for forecast members (each dims: time,y,x)")
    p.add_argument("--var",       default="generated",
                   help="Variable name in all files")
    p.add_argument("--out",       default="crps.nc",
                   help="Output CRPS NetCDF")
    args = p.parse_args()

    # 1) Open observations
    obs = xr.open_dataset(args.obs, chunks={'time':100,'y':64,'x':64})['ST']
    print("OBS dims:", obs.dims, "shape:", obs.shape)

    # 2) Find all member files
    fnames = sorted(glob.glob(args.fc_pattern))
    if not fnames:
        raise FileNotFoundError(f"No files match {args.fc_pattern!r}")
    print(f"Found {len(fnames)} members")

    # 3) Lazy‐load each one, expand dims, collect
    members = []
    for idx, fn in enumerate(fnames):
        da = xr.open_dataset(fn, chunks={'time':100,'y':64,'x':64})[args.var]
        da = da.expand_dims(member=[idx])  # now dims: ('member', 'time','y','x')
        members.append(da)

    # 4) Concatenate into one Dask DataArray
    fc = xr.concat(members, dim='member')
    # Guarantee the ordering of dims
    fc = fc.transpose('member', 'time', 'y', 'x')

    print("FC dims:", fc.dims, "shape:", fc.shape)

    # 5) Force time‐coords to match exactly
    fc = fc.assign_coords(time=obs.time)

    # 1) Make sure the entire member axis is one chunk
    fc = fc.chunk({'member': -1})

    # 2) Compute CRPS, allowing any internal rechunking
    crps = xs.crps_ensemble(
        observations=obs,
        forecasts=fc,
        keep_attrs=True,
        dim = 'time'
    )
    print("CRPS dims:", crps.dims, "shape:", crps.shape)

    # 7) Write out (you can add encoding/chunks here if you like)
    crps.to_netcdf(args.out)
    print(f"Written CRPS to {args.out}")

if __name__ == "__main__":
    main()
