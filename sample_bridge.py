import xarray as xr
from src.configuration import parse_command_line
from src.sde_model.inference import Inference as SDEInference
from src.consistency_model.inference import ConsistencyInference as CMInference
from src.sde_model.evaluate import Experiment
import torch  
import numpy as np 
from src.configuration import Config
from src.utils.transforms import apply_transforms, apply_inverse_transforms
def main():
    """ Command line interface for generating samples. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    esm_initial_condition_ts = xr.open_dataset("/data/datasets/CMIP6/ISMIP7/TS/CESM2-WACCM_TS_1850-2300_qdm_TS.nc")['ST'][:,:,:]
    esm_initial_condition = xr.open_dataset("/data/datasets/CMIP6/ISMIP7/SMB/CESM2-WACCM_SMB_1850-2300_qdm.nc")['SMB'][:,:,:]
    
    esm_initial_condition = esm_initial_condition.where(
    esm_initial_condition <= 1e6,
    0)
    
    esm_initial_condition_ts = esm_initial_condition_ts.where(
    esm_initial_condition_ts <= 1e6,
    0)
    
    ### for constraining, the transform stats have to be loaded, since the constraining has to happen in real units

    config = parse_command_line()
    config.checkpoint_path = '/data/results_oversample_asinh'
    config.sample_dimension = [576,336]#[72, 48]

    # insolation maps loading + surf maps loading
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

    # for converting back to xarray DataArray: 
    inf.training_target = inf.test_input = xr.open_dataset('/data/datasets/CMIP6/SMB_ST_5km_MAR_merged_masked.nc')["SMB"][:] 
    inf.training_target = inf.training_target.sel(time=inf.training_target.time.dt.year.isin(config.train_years)) 
    
    inf.training_target_ts = xr.open_dataset('/data/datasets/CMIP6/SMB_ST_5km_MAR_merged_masked.nc')["ST"][:] 
    inf.training_target_ts = inf.training_target_ts.sel(time=inf.training_target_ts.time.dt.year.isin(config.train_years)) 
    
    esm_initial_condition_inv = apply_transforms(esm_initial_condition, data_ref = inf.training_target, config = config)[:,:,:]
    esm_initial_condition_inv_ts = apply_transforms(esm_initial_condition_ts, data_ref = inf.training_target_ts, config = config)[:,:,:]





    inf.load_model(checkpoint_fname=f'best_consistency_model.ckpt')

    for sample_times in [10]: #[0 to 80]:
        samples = inf.run_stroke_guidance(esm_initial_condition_inv,sample_times = np.array([sample_times]), convert_to_xarray=True, ts_map = esm_initial_condition_inv_ts, inverse_transform=True,
                        month=insolation_map, srf_map= srf_map, ins_map=insolation_map, 
                        )
     
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
        "esm":       canon(samples["esm"].astype("float32")),
        "conditions":canon(samples["conditions"].astype("float32")),
        "ts":        canon(samples["ts"].astype("float32")),
        "cond_ts":   canon(samples["cond_ts"].astype("float32")),
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


        ds.to_netcdf(f'/data/save_{sample_times}.nc')

if __name__ == "__main__":
    main()    
    

