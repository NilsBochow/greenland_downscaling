import xarray as xr
from src.configuration import parse_command_line
from src.sde_model.inference import Inference as SDEInference
from src.consistency_model.inference import ConsistencyInference as CMInference
from src.sde_model.evaluate import Experiment
import torch  
import numpy as np 

from src.configuration import Config
def main():
    """ Command line interface for generating samples. """

    config = parse_command_line()
    config.checkpoint_path = '/data/results_oversample_asinh' #
    config.sample_dimension = [576,336]

    ##insolation maps loading 
    data_path: str =  "/data/datasets/insolation_masked.nc"
    target = xr.open_dataset(data_path, use_cftime=True)["insolation"]
    target.astype('float32')
    target = (target - np.min(target)) / (np.max(target) - np.min(target))
    insolation_map = target * 2 - 1
    ### srf map loading 
    data_path: str =  "/data/datasets/SRF_masked.nc"
    target = xr.open_dataset(data_path, use_cftime=True)["SRF"][:]
    target.astype('float32')
    target = (target - np.min(target)) / (np.max(target) - np.min(target))
    srf_map = target * 2 - 1

    if config.diffusion_model == 've':
        inf = SDEInference(config)
    elif config.diffusion_model == 'consistency':
        inf = CMInference(config)

    data = np.genfromtxt("/data/TS_train_min_max_values.csv", delimiter=",", skip_header=1)
    # In case the file contains a single row of values,
    # data will be a 1D array with two elements: [min, max]
    if data.ndim == 1:
        min_val, max_val = data[0], data[1]
    else:
        # If for some reason the file has multiple rows, 
        min_val, max_val = data[0, 0], data[0, 1]

    ts_initial_condition = xr.open_dataset("/data/datasets/CMIP6/NorESM/SMB_ST_5km_MAR_NorESM_245_masked.nc") #xr.open_dataset("/data/datasets/quantile_mapping/NorESM_tas/NorESM-MM_585_ts_GrIS_quantile_mapping_seasonal_era2.nc")#("/data/datasets/CMIP6/NorESM/SMB_ST_5km_MAR_NorESM_245_masked.nc") #")
    ts_array = ts_initial_condition['ST'][0:12,:,:]

    #normalise: 
    ts_array = (ts_array - min_val)/(max_val - min_val + 0.0001)
    ts_array = ts_array*2 - 1

    # for converting back to xarray DataArray: 
    inf.training_target = inf.test_input = xr.open_dataset('/data/datasets/CMIP6/SMB_ST_5km_MAR_merged_masked.nc')["SMB"] #xr.open_dataset('/data/datasets/merged_40km.nc') 
    inf.training_target = inf.training_target.sel(time=inf.training_target.time.dt.year.isin(config.train_years)) 
    inf.training_target_ts = xr.open_dataset('/data/datasets/CMIP6/SMB_ST_5km_MAR_merged_masked.nc')["ST"] 
    inf.training_target_ts = inf.training_target_ts.sel(time=inf.training_target_ts.time.dt.year.isin(config.train_years)) 
    inf.load_model(checkpoint_fname=f'best_{config.diffusion_model}_model.ckpt')
    
    months = (np.ones(12)*int(0)) #np.arange(12)
    months = months.astype(int)
    samples = inf.run(convert_to_xarray=True,
                      inverse_transform=True,
                      month=months, ins_map =insolation_map, srf_map= srf_map, ts_map = ts_array)
   


    def canon(da):
            # 1) normalize dim names to time/y/x
            rename = {}
            for d in da.dims:
                if d in ("t",): rename[d] = "time"
                if d in ("lat","latitude","south_north","j","nj"): rename[d] = "y"
                if d in ("lon","longitude","west_east","i","ni"):  rename[d] = "x"
            da = da.rename(rename)

            # 2) ensure coords exist (helps many viewers)
            if "y" in da.dims and "y" not in da.coords:
                da = da.assign_coords(y=np.arange(da.sizes["y"]))
            if "x" in da.dims and "x" not in da.coords:
                da = da.assign_coords(x=np.arange(da.sizes["x"]))
            if "time" in da.dims and "time" not in da.coords:
                # if you have a real time vector, use that instead
                da = da.assign_coords(time=np.arange(da.sizes["time"]))

            # 3) put dims in (time, y, x) order (dropping any that don't exist)
            order = [d for d in ("time","y","x") if d in da.dims]
            return da.transpose(*order)

    ds = xr.Dataset({
        "generated": canon(samples["generated"].astype("float32")),
        "ts":        canon(samples["ts"].astype("float32")),
    })

    # optional but nice: CF-ish metadata for coords
    if "x" in ds.dims:
        ds["x"].attrs.update(dict(standard_name="projection_x_coordinate", units="1"))
    if "y" in ds.dims:
        ds["y"].attrs.update(dict(standard_name="projection_y_coordinate", units="1"))
    if "time" in ds.dims:
        ds["time"].attrs.update(dict(standard_name="time"))

    # make time unlimited so ncview uses it as the slider
    encoding = {}
    if "time" in ds.dims:
        encoding["time"] = {"unlimited": True}

    # clean empty attrs (as you already do)
    for var in ds.data_vars:
        if ds[var].attrs.get("description") is None:
            ds[var].attrs.pop("description", None)
    
    ds.to_netcdf('/sample_consistency.nc')


if __name__ == "__main__":
    main()