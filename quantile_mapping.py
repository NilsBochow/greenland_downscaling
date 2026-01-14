import xarray as xr
from src.configuration import parse_command_line
from src.sde_model.inference import Inference as SDEInference
from src.consistency_model.inference import ConsistencyInference as CMInference
from src.sde_model.evaluate import Experiment
import torch  
from src.utils.quantile_mapping import QuantileMapping

def main(): 

    model_path = "/data/datasets/CMIP6/ISMIP7/SMB/CESM2-WACCM_1850-2300_smb_masked.nc"
    target_path = "/concatenated_SMB_TS_T2M_ERA5-5km_masked.nc"
    output_path = "/data/datasets/CMIP6/ISMIP7/SMB/CESM2-WACCM_SMB_1950-2014_qdm.nc"
    qm = QuantileMapping(model_path=model_path,
    target_path=target_path,
    out_path=output_path)

    qm.load_data()
    qm.run()
    qm.save()
if __name__ == "__main__":
    main()