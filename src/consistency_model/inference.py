import torch
from tqdm import tqdm
import numpy as np 
from pathlib import Path
from diffusers import UNet2DModel
import xarray as xr
from src.sde_model.inference import Inference
from src.configuration import Config
from src.consistency_model.model import Consistency
from src.utils.transforms import apply_inverse_transforms
from typing import Optional
from dataclasses import dataclass

@dataclass
class TransformStats:
    ref_mean: torch.Tensor
    ref_std:  torch.Tensor
    ref_min:  torch.Tensor
    ref_max:  torch.Tensor
    log_mean: torch.Tensor
    log_std:  torch.Tensor

@dataclass
class AsinhStats:
    # required for asinh → standardize
    s0:    torch.Tensor        # [1,1,H,W] or [H,W]
    mu_z:  torch.Tensor        # [1,1,H,W] or [H,W]
    sig_z: torch.Tensor        # [1,1,H,W] or [H,W]
    # optional extras (if present in your stats file)
    mean_y: Optional[torch.Tensor] = None
    std_y:  Optional[torch.Tensor] = None
    lo_y:   Optional[torch.Tensor] = None
    hi_y:   Optional[torch.Tensor] = None
     # ---- Back-compat aliases (old code expects these) ----
    @property
    def ref_mean(self):   # used where 'standardize' expects a mean
        return self.mu_z

    @property
    def ref_std(self):    # used where 'standardize' expects a std
        return self.sig_z

    @property
    def log_mean(self):   # if any leftover code still refers to log_* names
        return self.mu_z

    @property
    def log_std(self):
        return self.sig_z
    
    @property
    def ref_min(self):
        return self.lo_y
    
    @property
    def ref_max(self):
        return self.hi_y


class ConsistencyInference(Inference):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.config = config
        self.transform_stats_smb = self.load_asinh_stats(
            "/data/datasets/CMIP6/transform_stats_train_MAR_SMB_asinh.nc", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.transform_stats_ts = self.load_asinh_stats(
            "/data/datasets/CMIP6/transform_stats_train_MAR_ST_asinh.nc", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    def load_model(self,
                   checkpoint_fname: str = 'best'):
        """Loads the model from a checkpoint.

            checkpoint_fname: Path to the .ckpt file. Default 'best' loads .../best_model.ckpt
        """

        if checkpoint_fname == 'best':
            self.checkpoint_path = f'{self.config.checkpoint_path}/best_model.ckpt'
        else:
            self.checkpoint_path = f'{self.config.checkpoint_path}/{checkpoint_fname}'
        assert Path(self.checkpoint_path).exists(), f"Path {self.checkpoint_path} does not exist."

        self.checkpoint = torch.load(self.checkpoint_path)

        model_hyperparameters = ['channels', 'down_block_types', 'up_block_types', 'diffusion_model',
                                 'sigma', 'sigma_max', 'sigma_min', 'epsilon']

        config_checkpoint = {}
        for key in self.checkpoint['hyper_parameters'].keys(): 

            if key in model_hyperparameters:
                setattr(self.config, key, self.checkpoint['hyper_parameters'][key])

            config_checkpoint[key] = self.checkpoint['hyper_parameters'][key]

        network = UNet2DModel(
            in_channels=1,
            out_channels=1,
            block_out_channels=(128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.model = Consistency.load_from_checkpoint(model=network,
                                         checkpoint_path=self.checkpoint_path,
                                         config=self.config)

        self.model.config_checkpoint = config_checkpoint
        self.model.transform_stats_smb = self.transform_stats_smb
        self.model.transform_stats_ts  = self.transform_stats_ts

        self.model.to(self.device)
        self.model.eval()



    def load_transform_stats(self, path: str, device: torch.device):
        """
        Load normalization and log‐stats from a zarr/NetCDF file and
        return them as torch.FloatTensors on `device`.
        """
        ds = xr.open_dataset(path)
        def _to_t(name):
            return torch.from_numpy(ds[name].values).to(device=device, dtype=torch.float32)

        ref_mean = _to_t("mean")
        ref_std  = _to_t("std")
        ref_min  = _to_t("lo")
        ref_max  = _to_t("hi")
        log_mean = _to_t("log_mean")
        log_std  = _to_t("log_std")

        return ref_mean, ref_std, ref_min, ref_max, log_mean, log_std

    def load_asinh_stats(self, path: str, device: torch.device) -> AsinhStats:
        ds = xr.open_dataset(path)

        def _to_t(name):
            return torch.from_numpy(ds[name].values).to(device=device, dtype=torch.float32)

        # required
        s0    = _to_t("s0")      # [H,W]
        mu_z  = _to_t("mu_z")    # [H,W]
        sig_z = _to_t("sig_z")   # [H,W]

        # optional extras if present
        mean_y = _to_t("mean_y") if "mean_y" in ds else None
        std_y  = _to_t("std_y")  if "std_y"  in ds else None
        lo_y   = _to_t("lo_y")   if "lo_y"   in ds else None
        hi_y   = _to_t("hi_y")   if "hi_y"   in ds else None

        # make broadcast-friendly: [1,1,H,W]
        def _expand(t): return None if t is None else t.unsqueeze(0).unsqueeze(0)
        return AsinhStats(
            s0=_expand(s0),
            mu_z=_expand(mu_z),
            sig_z=_expand(sig_z),
            mean_y=_expand(mean_y),
            std_y=_expand(std_y),
            lo_y=_expand(lo_y),
            hi_y=_expand(hi_y),
        )

    def run(self,
            convert_to_xarray: bool=True,
            inverse_transform: bool=True,
            num_samples: int = 12,
            steps: int = 10,
            use_ema: bool = False,
            srf_map: Optional[torch.Tensor] = None,
            month: Optional[torch.Tensor] = None,
            ins_map: Optional[torch.Tensor] = None,
            ts_map: Optional[torch.Tensor] = None
            ) -> dict:
        """Executes the inference sampling unconditionally from the learned distribution.

        Args:
            convert_to_xarray: Convertes torch tensor results to xarray dataset.
            inverse_transform: Transform results back to phyical space.
            num_samples: Number of generated samples per batch.
            steps: Number of integration steps.
            use_ema: Enables exponential moving average model 

        Returns:
            Dictionary containing generated samples
        """
 
        all_samples = []
        all_samples_ts = []
        def convert_to_tensor(da):
            if da is not None:
                return torch.from_numpy(da.values).float().to(self.model.device)
            return None

        
        if month is not None:
            # Convert month to a torch tensor without adding extra dimensions
            month_tensor = torch.as_tensor(month, device=self.model.device)
            print("month_tensor shape", month_tensor.shape, month_tensor.dim())
            # If month_tensor is a scalar (0-dim), make it 1-dimensional
            #if month_tensor.dim() == 0:
            month_tensor = month_tensor.unsqueeze(0)
            #else:
            #    month_tensor = None

        # Process ins_map using month index
        ins_tensor = None
        if ins_map is not None and month is not None:
            # Select the specific month's data
            print("month ins map", month)
            ins_data = ins_map[month,:,:]
            print("ins_data", ins_data)
            # Convert to tensor and add dimensions
            ins_tensor = torch.from_numpy(ins_data.values).float().to(self.model.device)
            #ins_tensor = ins_tensor.unsqueeze(0)  # Add batch dimension
            #ins_tensor = ins_tensor.repeat(num_samples, 1, 1)  # Match batch size



        srf_tensor = torch.from_numpy(srf_map.values).float().to(self.model.device)
        if srf_tensor.dim() == 2:
            srf_tensor = srf_tensor.unsqueeze(0)  # Add batch dimension

        if ins_tensor.dim() == 2:
            ins_tensor = ins_tensor.unsqueeze(0)  # Add batch dimension

        # Validate SRF map count
        total_samples_needed = self.config.num_batches * num_samples
        #if srf_tensor.shape[0] not in (1, total_samples_needed):
        #    raise ValueError(f"SRF map count {srf_tensor.shape[0]} must be 1 or match total samples {total_samples_needed}")
        print("shapes 0", month_tensor.shape, ins_tensor.shape, srf_tensor.shape)
        # Repeat if using single SRF map
        if srf_tensor.shape[0] == 1:
            srf_tensor = srf_tensor.repeat(num_samples, 1, 1)

        print("shapes ", month_tensor.shape, ins_tensor.shape, srf_tensor.shape)
        for b in tqdm(range(self.config.num_batches)):
            ts_tensor = None 
            if ts_map is not None and month is not None:
                # Select the specific month's data
                print(b*12, 12*(b+1), ts_map.shape)
                ts_tensor = ts_map[b*12 : 12*(b+1),:,:]
                # Convert to tensor and add dimensions
                ts_tensor = torch.from_numpy(ts_tensor.values).float().to(self.model.device)

                #ts_tensor = ts_tensor.repeat(num_samples, 1, 1)  # Match batch size
            print("shapes 2", month_tensor.shape, ins_tensor.shape, srf_tensor.shape, ts_tensor.shape)
            print(month_tensor)
            samples, samples_ts = self.model.sample(
                                        num_samples = num_samples,
                                        steps = steps,
                                        x_image_size = self.config.sample_dimension[0],
                                        y_image_size = self.config.sample_dimension[1],
                                        use_ema = use_ema,
                                        month= month_tensor,
                                        srf_map= srf_tensor,
                                        ins_map = ins_tensor,
                                        )
            all_samples.append(samples)
            all_samples_ts.append(samples_ts)

        all_samples = torch.cat(all_samples).cpu().numpy()
        all_samples_ts= torch.cat(all_samples_ts).cpu().numpy()
        if all_samples.shape[-2] == 64:
            all_samples = all_samples[:,:,2:62] # remove padding

        if convert_to_xarray:
            all_samples = self.convert_to_xarray(all_samples)
            all_samples_ts =  self.convert_to_xarray(all_samples_ts)

        else:
            all_samples = all_samples


        if inverse_transform:
            data = np.genfromtxt("/data/TS_train_min_max_values.csv", delimiter=",", skip_header=1)
            if data.ndim == 1:
                min_val, max_val = data[0], data[1]
            else:
                min_val, max_val = data[0, 0], data[0, 1]
            all_samples = apply_inverse_transforms(all_samples,
                                                   config=self.config, data_ref = self.training_target,
                                                    ref_s0=self.transform_stats_smb.s0,
                                                    ref_mean=self.transform_stats_smb.mean_y,
                                                    ref_std=self.transform_stats_smb.std_y
                                                    )
            all_samples_ts = apply_inverse_transforms(all_samples_ts, config=self.config, data_ref = self.training_target_ts,
                                                    ref_s0=self.transform_stats_ts.s0,
                                                    ref_mean=self.transform_stats_ts.mean_y,
                                                    ref_std=self.transform_stats_ts.std_y)
                                                    
    
        return {'generated': all_samples, 'ts': all_samples_ts}



    def run_stroke_guidance(self,
            esm_dataloader,
            sample_times,
            convert_to_xarray: bool=True,
            inverse_transform: bool=True,
            num_samples = 1,
            num_batches = 1,
            steps = 10,
            use_ema=True,
            month: Optional[torch.Tensor] = None,
            srf_map: Optional[torch.Tensor] = None,
            ins_map: Optional[torch.Tensor] = None,
            ts_map: Optional[torch.Tensor] = None,               # ← new    
            ) -> dict:
        """Executes the inference sampling by noising an upsampled ESM field.

        Returns a dict with keys:
        - 'generated': image‐samples
        - 'conditions': (existing) second output of sample_conditional
        - 'ts': the TS‐samples
        - 'esm': the raw ESM fields
        """

        all_esm = []
        all_samples = []
        all_conditions = []    # your existing second output
        all_ts = []            # ← collect the new TS‐samples
        all_Ts_conditions = []
        def convert_to_tensor(da):
            if da is not None:
                return torch.from_numpy(da.values).float().to(self.model.device)
            return None

        # pre‐convert all static maps
        month_tensor = convert_to_tensor(month)

        srf_tensor = convert_to_tensor(srf_map)
        if srf_tensor is not None and srf_tensor.dim() == 2:
            srf_tensor = srf_tensor.unsqueeze(0)
        if srf_tensor is not None and srf_tensor.shape[0] == 1:
            srf_tensor = srf_tensor.repeat(num_samples, 1, 1)

        ins_tensor = convert_to_tensor(ins_map)
        if ins_tensor is not None and ins_tensor.dim() == 2:
            ins_tensor = ins_tensor.unsqueeze(0)

        ts_map = convert_to_tensor(ts_map)  

        total_samples_needed = self.config.num_batches * num_samples
        index = 0
        months = esm_dataloader.time.dt.month.values

        for x in tqdm(esm_dataloader):
            x = torch.from_numpy(x.values)
            for b in range(num_batches):
                # pick out per‐batch conditioning
                month_idx = months[index] - 1
                init_x = x.to(self.device).float().unsqueeze(0)
                ts_tensor = ts_map[index]
                print("ts_tensor shape", ts_tensor.shape)
                # if your ins_map is 12×H×W, pick the per‐month slice
                ins_tensor_single = ins_tensor[month_idx] if ins_tensor is not None else None

                samples, cond, ts_cond, ts_samples = self.model.sample_conditional(
                    init_x,
                    init_x.shape[-2],
                    init_x.shape[-1],
                    steps = steps,
                    sample_times = sample_times,
                    use_ema = use_ema,
                    month = month_tensor,
                    srf_map = srf_tensor,
                    ins_map = ins_tensor_single,
                    ts_map = ts_tensor,                   
                    constraints = True,
                    transform_stats_smb = self.transform_stats_smb,
                    transform_stats_ts = self.transform_stats_ts
                )
                print("samples ", samples )
                all_esm.append(init_x.cpu())
                all_samples.append(samples.cpu())
                all_conditions.append(cond.cpu())
                all_ts.append(ts_samples.cpu())     
                all_Ts_conditions.append(ts_cond.cpu())       # ← store it

            index += 1

        all_esm       = torch.cat(all_esm)
        all_samples   = torch.cat(all_samples)
        all_conditions= torch.cat(all_conditions)
        all_ts        = torch.cat(all_ts)                  # ← concat TS
        all_Ts_conditions = torch.cat(all_Ts_conditions)
        if convert_to_xarray:
            all_esm        = self.convert_to_xarray(all_esm.numpy())
            all_samples    = self.convert_to_xarray(all_samples.numpy())
            all_conditions = self.convert_to_xarray(all_conditions.numpy())
            all_ts         = self.convert_to_xarray(all_ts.numpy())
            all_Ts_conditions  = self.convert_to_xarray(all_Ts_conditions.numpy())
        print("all samples", all_samples)
        if inverse_transform:
            all_ts = apply_inverse_transforms(all_ts,        config=self.config,
        ref_s0=self.transform_stats_ts.s0, data_ref = self.training_target_ts,
        ref_mean=self.transform_stats_ts.mean_y,
        ref_std=self.transform_stats_ts.std_y)
            all_esm        = apply_inverse_transforms(all_esm,        config=self.config,
        ref_s0=self.transform_stats_smb.s0, data_ref = self.training_target,
        ref_mean=self.transform_stats_smb.mean_y,
        ref_std=self.transform_stats_smb.std_y)
            all_Ts_conditions = apply_inverse_transforms(all_Ts_conditions,        config=self.config,
        ref_s0=self.transform_stats_ts.s0, data_ref = self.training_target_ts,
        ref_mean=self.transform_stats_ts.mu_z,
        ref_std=self.transform_stats_ts.std_y)
    
            all_samples    = apply_inverse_transforms(all_samples,    config=self.config,
        ref_s0=self.transform_stats_smb.s0, data_ref = self.training_target,
        ref_mean=self.transform_stats_smb.mean_y,
        ref_std=self.transform_stats_smb.std_y)
            all_conditions = apply_inverse_transforms(all_conditions, config=self.config,
        ref_s0=self.transform_stats_smb.s0, data_ref = self.training_target,
        ref_mean=self.transform_stats_smb.mean_y,
        ref_std=self.transform_stats_smb.std_y)
        print("all samples 2", np.min(all_samples), np.max(all_samples))
        return {
            'generated': all_samples,
            'conditions': all_conditions,
            'ts': all_ts,                  # ← the new key
            'esm': all_esm,
            'cond_ts': all_Ts_conditions
        }

    def convert_to_xarray(self, samples: np.ndarray) -> xr.DataArray:
        """Transforms the samples tensor to xarray format."""

        print("samples.shape", samples.shape, len(samples.shape))

        # If samples have 4 dimensions (e.g., (time, channels, height, width))
        if len(samples.shape) == 4:
            # Keep only the first channel (if channels exist)
            samples = samples[:, 0, :, :]
            print("After removing channels, samples.shape", samples.shape, len(samples.shape))

        # Assuming the length of time should match samples' time dimension
        time_length = len(samples)

        # Create xarray DataArray
        results = xr.DataArray(
            data=samples,
            dims=["time", "y", "x"],  # The names of the spatial dimensions
            coords=dict(
                time=np.arange(samples.shape[0]),  # Match length with samples
                latitude=self.test_input.y,  # Dynamically choose lat or y
                longitude=self.test_input.x  # Dynamically choose lon or x
            ),
            attrs=dict(
                description=self.config.predict_variable  # Add description
            )
        )

        return results
