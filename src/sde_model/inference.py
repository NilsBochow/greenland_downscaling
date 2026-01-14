import torch
import numpy as np
import xarray as xr
from pathlib import Path
#from tqdm.notebook import tqdm
from tqdm import tqdm
from typing import List, Optional, Type, Union, Tuple
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
from dataclasses import replace
from src.utils.transforms import apply_inverse_transforms, apply_transforms
from src.sde_model.model import SDEModel
from src.configuration import Config
import src.utils.xarray_utils as xu

class Inference:
    def __init__(self,
                 config: Config):
        """Evaluates the trained score model.
        
        Args:
            config: Model configuration.
        """
        
        self.config = config
        self.ts_config = replace(self.config, transforms=['normalize_minus1_to_plus1'])
        self.batch_size = config.batch_size
        self.sample_dimension = config.sample_dimension 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.results = None
        self.checkpoint = None
        self.training_target = None
        self.test_target = None
        self.test_input = None
        
        
    def load_model(self,
                   checkpoint_fname: str = 'best'):
        """ Loads the model from a checkpoint.

        Args:
            checkpoint_fname: Path to the .ckpt file. Default 'best' loads .../best_model.ckpt
        """

        if checkpoint_fname == 'best':
            self.checkpoint_path = f'{self.config.checkpoint_path}/best_model.ckpt'
        else:
            self.checkpoint_path = f'{self.config.checkpoint_path}/{checkpoint_fname}'
        assert Path(self.checkpoint_path).exists(), f"Path {self.checkpoint_path} does not exist."

        self.checkpoint = torch.load(self.checkpoint_path)

        print("checkpoint_path inference: ", self.checkpoint_path)

        model_hyperparameters = ['channels', 'down_block_types', 'up_block_types', 'diffusion_model',
                                 'sigma', 'sigma_max', 'sigma_min', 'epsilon']

        config_checkpoint = {}
        for key in self.checkpoint['hyper_parameters'].keys(): 

            if key in model_hyperparameters:
                setattr(self.config, key, self.checkpoint['hyper_parameters'][key])

            config_checkpoint[key] = self.checkpoint['hyper_parameters'][key]
            
        self.model = SDEModel.load_from_checkpoint(self.checkpoint_path,
                                                   config=self.config)

        self.model.config_checkpoint = config_checkpoint
        self.model.to(self.device)
        self.model.eval()


    def load_data(self,
                      training_target: xr.DataArray,
                      test_target: xr.DataArray,
                      test_input: xr.DataArray):
                      #ts_target: Optional[xr.DataArray]):
        """ Loads datasets.
        
        Args:
            training_target: The ground truth training dataset.
            test_target: The target test set for comparisons.
            test_input: The input to be downscaled, e.g. the ESM test set.
        """

        self.training_target = training_target
        print("load_data inference.py", np.min(self.training_target), np.max(self.training_target), np.mean(self.training_target))
        self.test_target = test_target
        self.test_input = test_input
        #self.ts_target = ts_target

        
        
    def run(self,
            sampler_type="sde",
            num_steps=500,
            num_batches=12,
            convert_to_xarray=True,
            inverse_transform=True,
            show_progress=False, 
            months = 0,
            insolation_map = None,
            srf_map = None,
            ts = None,
            init_x: Optional[torch.tensor]=None) -> np.ndarray:
        """Executes the inference sampling.
        
        Args:
            sampler_type: "sde" 
            num_steps: Number of integration steps
            convert_to_xarray: Converts torch tensor to xarray's format
            inverse_transform: Either "sde" or "ode"
            show progress: Displays a progress bar
            init_x: Initial condition for the SDE, randomly generated if None is provided 

        Returns:
            The downscaled fields in physical units.
        """
        
        all_samples = []
        assert sampler_type in ['sde', 'ode', 'ode_gpu', 'pc'], "sampler type {sampler_type} can be sde, ode or pc."

        num_batches = range(num_batches)
        srf_map_tensor = torch.tensor(srf_map.values, device=self.device)  # Move to GPU if needed
        # Perform NumPy operations (if required)
        srf_map_cpu = srf_map_tensor.cpu()  # Move to CPU for NumPy operations
        

        if show_progress:
            num_batches = tqdm(num_batches)

        for i in num_batches:
            months = np.mod(i,12) 
            if sampler_type == 'sde':
                print(months)
                plt.imshow(insolation_map[months].values, vmin=0, vmax=1)
                plt.colorbar()
                plt.savefig(f"/p/projects/ou/labs/ai/Nils/arctic-downscaling/data/plots/insolation_month{months}.png")
                plt.clf()
                plt.imshow(srf_map_cpu, vmin=0, vmax=1)
                plt.colorbar()
                plt.savefig(f"/p/projects/ou/labs/ai/Nils/arctic-downscaling/data/plots/srfmap_month{months}.png")
                plt.clf()
                insolation_maps = torch.tensor(np.tile(insolation_map[months].values,(int(self.batch_size), 1, 1)), 
                                    dtype=torch.float32,
                                    device=self.device) 

                srf_map_tiled = np.tile(srf_map_cpu.numpy(), (int(self.batch_size), 1, 1))
                \
                # Convert back to PyTorch tensor (if needed)
                srf_map_tiled_tensor = torch.tensor(srf_map_tiled, device=self.device)
                

                #ts_batch = ts[i:i+1,:,:].to(self.device)

                samples = self.model.euler_maruyama_sampler(batch_size=self.batch_size,
                                                            sample_dimension=self.sample_dimension, 
                                                            init_x=init_x,
                                                            num_steps=num_steps,
                                                            month=insolation_maps,
                                                            srf_map=srf_map_tiled_tensor,
                                                            ts = None)

            all_samples.append(samples)

        all_samples = torch.cat(all_samples).cpu().numpy()
        print("samples.shape", all_samples.shape)
        if all_samples.shape[-2] == 64:
            all_samples = all_samples[:,:,2:62] # remove padding
        if all_samples.shape[-1] == 48:
            all_samples = all_samples[:,:,:,3:-3] # remove padding
        if convert_to_xarray:
            all_samples = self.convert_to_xarray(all_samples)

        if inverse_transform:
            all_samples = apply_inverse_transforms(all_samples,
                                                   self.training_target,
                                                   self.config)
        self.results = all_samples 
        return self.results


    def run_bridge(self,
                   esm_dataloader,
                   reverse_num_steps: int=500,
                   forward_num_steps: int=500,
                   num_batches=43,
                   stop_step: int=np.inf,
                   months = 0,
                   insolation_map = None,
                   srf_map = None,
                   ts = None,
                   convert_to_xarray: bool=True,
                   inverse_transform: bool=True) -> np.ndarray:
        """Executes the inference sampling by noising an upsampled ESM field and
        then denoising it with a reverse SDE.
        
        Args:
            forward_num_steps: number of steps for forward integration
            reverse_num_steps: number of steps for reverse integration
            
        Returns:
            The downscaled fields in physical units, the noised and raw ESM fields.
        """
 
        all_esm = []
        all_samples = []
        all_conditions = []
        
        #if stop_step is None:
        #        stop_step = max(forward_num_steps, reverse_num_steps)
        #print("esm_dataloader.shape", esm_dataloader.shape)
        #esm_dataloader = esm_dataloader.reshape(num_batches, -1, *esm_dataloader.shape[2:])
        #print("esm_dataloader.shape after reshaping", esm_dataloader.shape)
        #for b in range(num_batches):

        

        for batch_idx, x in enumerate(esm_dataloader):
           
            if isinstance(x, (list, tuple)):
                x = x[0]
            init_x = x.to(self.device)
            if len(init_x.shape) == 3:
                init_x = init_x.unsqueeze(1)

            # Calculate slice indices for months
            batch_size = init_x.shape[0]
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            # Extract months for this batch
            batch_months = months[start_idx:end_idx]  # Shape: (batch_size,)
            batch_months = torch.tensor(batch_months, dtype=torch.long).to(self.device)
            print(batch_months)
            batch_insolation = []
            for month in batch_months.cpu().numpy():
                # Get insolation map for this month (0-based vs 1-based indexing)
                insolation_maps = insolation_map[month].values
                plt.imshow(insolation_maps, vmin=0, vmax=1)
                plt.colorbar()
                plt.savefig(f"/p/projects/ou/labs/ai/Nils/arctic-downscaling/data/plots/insolation_month{month}.png")
                plt.clf()
                batch_insolation.append(insolation_maps)

            batch_insolation = torch.tensor(np.stack(batch_insolation), 
                                      dtype=torch.float32,
                                      device=self.device)

            srf_map_tensor = torch.tensor(srf_map.values, device=self.device)  # Move to GPU if needed
            # Perform NumPy operations (if required)
            srf_map_cpu = srf_map_tensor.cpu()  # Move to CPU for NumPy operations
            srf_map_tiled = np.tile(srf_map_cpu.numpy(), (int(end_idx - start_idx), 1, 1))

            # Convert back to PyTorch tensor (if needed)
            srf_map_tiled_tensor = torch.tensor(srf_map_tiled, device=self.device)
         
            init_x = x.to(self.device)

            ts_batch = ts[start_idx:end_idx].to(self.device)
            print("ts_batch.shape", ts_batch.shape)
            print("init_x.shape run brige", init_x.shape)
            if len(init_x.shape) == 3:
                init_x = init_x.unsqueeze(1)
            conditionings = self.model.conditional_euler_maruyama_sampler(batch_size=init_x.shape[0],
                                                                    sample_dimension=(init_x.shape[-2],init_x.shape[-1]),
                                                                    init_x=init_x,
                                                                    num_steps=forward_num_steps,
                                                                    stop_step=stop_step,
                                                                    forward=True,
                                                                    month=batch_insolation,
                                                                    srf_map=srf_map_tiled_tensor,
                                                                    ts = ts_batch)
            
            samples = self.model.conditional_euler_maruyama_sampler(batch_size=init_x.shape[0],
                                                                    sample_dimension=(init_x.shape[-2],init_x.shape[-1]),
                                                                    init_x=conditionings,
                                                                    num_steps=reverse_num_steps,
                                                                    stop_step=stop_step,
                                                                    forward=False,
                                                                    month=batch_insolation,
                                                                    srf_map=srf_map_tiled_tensor,
                                                                    ts = ts_batch)
            all_esm.append(init_x)
            all_samples.append(samples)
            all_conditions.append(conditionings)

        all_esm = torch.cat(all_esm).cpu().numpy()
        all_samples = torch.cat(all_samples).cpu().numpy()
        all_conditions = torch.cat(all_conditions).cpu().numpy()
        all_ts = (ts).cpu().numpy()

        if all_esm.shape[-2] == 64:
            all_esm = all_esm[:,:,2:62,:] # remove padding
            all_samples= all_samples[:,:,2:62,:] # remove padding
            all_conditions = all_conditions[:,:,2:62,:] # remove padding

        if convert_to_xarray:
            all_esm = self.convert_to_xarray(all_esm)
            all_samples = self.convert_to_xarray(all_samples)
            all_conditions = self.convert_to_xarray(all_conditions)
            all_ts = self.convert_to_xarray(all_ts)
        else:
            all_esm = all_esm
            all_samples = all_samples
            all_conditions = all_conditions

        if inverse_transform:
            all_esm = apply_inverse_transforms(all_esm,
                                               self.training_target,
                                               self.config)
            all_samples = apply_inverse_transforms(all_samples,
                                                   self.training_target,
                                                   self.config)
            all_conditions = apply_inverse_transforms(all_conditions,
                                                   self.training_target,
                                                   self.config)
            all_ts = apply_inverse_transforms(all_ts,
                                                   self.ts_target,
                                                   self.ts_config)
            print("run_bidge", all_esm.shape, all_samples.shape, all_conditions.shape, all_ts.shape)
        return {'generated': all_samples, 'conditions': all_conditions, 'esm': all_esm, 'ts': all_ts}
    
    
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

    def convert_to_xarray_conditions(self, conditions: np.ndarray) -> xr.DataArray:
        """Transforms the conditions tensor to xarray format."""

        print("conditions.shape", conditions.shape, len(conditions.shape))

        # If conditions have 4 dimensions (e.g., (time, channels, height, width))
        if len(conditions.shape) == 4:
            # Keep only the first channel (if channels exist)
            conditions = conditions[:, 0, :, :]
            print("After removing channels, conditions.shape", conditions.shape, len(conditions.shape))

        # Assuming the length of time should match conditions' time dimension
        time_length = len(conditions)

        # Create xarray DataArray
        results = xr.DataArray(
            data=conditions,
            dims=["time", "y", "x"],  # The names of the spatial dimensions
            coords=dict(
                time=self.test_input.time[:time_length],  # Match length with conditions
                latitude=self.test_input.y,  # Dynamically choose lat or y
                longitude=self.test_input.x  # Dynamically choose lon or x
            ),
            attrs=dict(
                description=self.config.predict_variable  # Add description
            )
        )

        return results

