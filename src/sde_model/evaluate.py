from typing import List, Optional, Type, Union, Tuple
import numpy as np
import torch 
import pickle
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import replace
from src.sde_model.inference import Inference
from src.data import GeoDataset
import src.utils.xarray_utils as xu
from src.utils.transforms import apply_transforms
from src.configuration import Config
from src.utils.spectra import mean_rapsd
from src.utils.utils import time_to_steps


class Experiment():
    
    def __init__(self,
                 config: Config,
                 num_sde_steps: Optional[int]=500) -> None:
        """ Collects methods for data and model loading, inference and plotting. """
        
        self.config = config
        self.samples = {}
        self.num_sde_steps = num_sde_steps
        self.inference = Inference(config)
        self.conditionings = {}
        self.ts_inverse = {}

    def prepare_data(self, lazy: Optional[bool]=True) -> None:
        """ Load datasets from file and preprocesses them."""

        self.config.lazy = lazy
        print("prepare_data 1 ")
        self.era5_train = GeoDataset("train", "ERA5", self.config).target.astype(np.float32)
        self.era5_test = GeoDataset("test", "ERA5", self.config).target.astype(np.float32)
        print("prepare_data 2 ")
        
        self.esm_train = GeoDataset("train", "ESM", self.config).climate_model.astype(np.float32)
        self.esm_test = GeoDataset("test", "ESM", self.config).climate_model.astype(np.float32)
        #print("prepare data, esm", self.esm_train.shape, self.esm_test.shape)
        #self.ts_train_target = GeoDataset("train", "ERA5", self.config).target_reference_ts.astype(np.float32)
        print("prepare_data 3 ")
        """
        self.inference.load_data(training_target=self.era5_train,
                       test_target=self.era5_test,
                       test_input=self.esm_test,
                       ts_target = self.ts_train_target)
        """
        self.inference.load_data(training_target=self.era5_train,
                test_target=self.era5_test,
                test_input=self.esm_test)



    def load_model(self, checkpoint_fname: str) -> None:
        """ Load model checkpoint from file. """

        self.inference.load_model(checkpoint_fname=checkpoint_fname)
        
        
    def sample_unconditional(self, show_progress: Optional[bool]=True) -> None:
        """ Generates unconditional samples. """

        self.inference.sample_dimension = (len(self.era5_test.latitude.values), len(self.era5_test.longitude.values))
    
        self.samples['unconditional'] = self.inference.run(sampler_type='sde',
                                                           num_steps=self.num_sde_steps,
                                                           convert_to_xarray=True,
                                                           inverse_transform=True,
                                                           show_progress=show_progress)
        
        
    def transform_initial(self, esm_initial_condition, ts_esm):
        self.esm_initial_condition = apply_transforms(esm_initial_condition, data_ref=self.era5_train, config=self.config)
        modified_config = replace(self.config, transforms=['normalize_minus1_to_plus1'])
        #self.ts_esm = apply_transforms(ts_esm, data_ref=self.ts_train_target, config=modified_config)
        #return self.ts_esm

        

    def sample_bridge(self,
                        noise_times: list,
                        batch_size,
                        esm_initial_condition: torch.tensor,
                        insolation_map: torch.tensor, 
                        srf_map : torch.tensor,
                        months) -> None:
            """ Uses the SDE brige for sampling.

            Args:
                noise_times: list of times to terminate the forward SDE.
                esm_initial_condition: torch tensor containing ESM fields
                interpolated to the diffusion model resolution of the shape [batch, channel, height, width]
            """
            print("sample bridge shape", self.esm_initial_condition.shape)
            self.esm_initial_condition =  TensorDataset(torch.tensor(self.esm_initial_condition.values).float())#.unsqueeze(0)
            self.ts_esm = torch.tensor(self.ts_esm.values).float()
            esm_dataloader = DataLoader(self.esm_initial_condition, batch_size=batch_size, shuffle=False, drop_last=False)
            for batch in esm_dataloader:
                print(batch[0].shape)  # Should print (32, 1, 64, 64) for most batches

            for noise_time in noise_times:

                stop_step = time_to_steps(noise_time, self.num_sde_steps)

                results = self.inference.run_bridge(esm_dataloader=esm_dataloader,
                                                    reverse_num_steps=self.num_sde_steps,
                                                    forward_num_steps=self.num_sde_steps,
                                                    stop_step=stop_step,
                                                    convert_to_xarray=True,
                                                    inverse_transform=True, 
                                                    months = months,
                                                    insolation_map = insolation_map,
                                                    srf_map = srf_map,
                                                    ts = None)
                self.conditionings[noise_time] = results['conditions']
                self.samples[noise_time] = results
                #self.ts_inverse[noise_time] = results['ts']
                print("results shape", results)
                print("self.samples.shape", self.samples)
            return self.samples
                
    def plot_sample(self) -> None:
        """ Plots a generated and target sample. """

        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.title("Sample (uncond.)")
        plt.imshow(self.samples['unconditional'][0], origin='lower', vmax=0.0002)
        
        plt.subplot(1,2,2)
        plt.title("Target")
        plt.imshow(self.era5_test[0], origin='lower', vmax=0.0002)

        plt.show()

    
    def save(self, fname: str) -> None:
        """ Saves dictionary with results to disk. """

        for key in self.samples.keys():
            self.samples[key].load()

        with open(fname, 'wb') as f:
            pickle.dump(self.samples, f)

    def save_all_netcdf(self, base_fname: str) -> None:
        """ Saves all dictionary entries (xarray datasets) to disk as netcdf, including conditionings. """
        
        if not self.samples:
            print("self.samples is empty.")
            return

        for key in self.samples.keys():
            print(f"Saving data for key: {key}")

            # Retrieve and rename the generated data
            generated_data = self.samples[key]['generated'].rename("SMB")
            print("generated_data shape, save_netcdf", generated_data)

            # Retrieve and rename the conditionings data
            if isinstance(self.conditionings[key], xr.DataArray):
                conditionings_data = self.conditionings[key].rename("Conditionings")
            else:
                conditionings_data = self.conditionings[key]
            if isinstance(self.conditionings[key], xr.DataArray):
                ts = self.ts_inverse[key].rename("TS")
            else:
                ts = self.ts_inverse[key]
            
            # Combine the generated data and conditionings into a single dataset
            combined_data = xr.merge([generated_data, conditionings_data, ts])

            # Construct the filename for each key
            fname = f"{base_fname}_{key}.nc"

            # Write the combined dataset to a NetCDF file
            xu.write_dataset(combined_data, fname)
            print(f"Saved {fname}")
    def save_netcdf(self, fname: str, key=None) -> None:
        """ Saves a single dictionary entry (xarray dataset) to disk as netcdf, including conditionings. """
        
        if self.samples:
            print("Keys in self.samples:")
            for key in self.samples.keys():
                print(key)
        else:
            print("self.samples is empty.")
        
        if key is None:
            generated_data = self.samples['unconditional'].rename("SMB")
        else:
            generated_data = self.samples[key]['generated'].rename("SMB")
        print("generated_data shape, save_netcdf", generated_data)
        # Assuming self.conditionings is an xarray.Dataset or can be converted to one
        if isinstance(self.conditionings, xr.DataArray):
            conditionings_data = self.conditionings[key].rename("Conditionings")
        else:
            conditionings_data = self.conditionings
        
        # Combine the generated data and conditionings into a single dataset
        combined_data = xr.merge([generated_data, conditionings_data])
        
        # Write the combined dataset to a NetCDF file
        xu.write_dataset(combined_data, fname)

            
            
    def load(self, fname: str) -> None:
        """ Loads saved dictionary from a given file. """

        with open(fname, 'rb') as handle:
            self.samples = pickle.load(handle)


    def compute_spectra(self) -> None:
        """ Computes radially averaged power spectral densities of:
            - the ESM data
            - the ERA5 target data.
        """
        
        #num_latitudes = len(self.era5_test.y)
        #offset = num_latitudes//2
        print("compute spectra, self.era5_test", self.era5_test.shape)
        era5 = apply_transforms(self.era5_test[:,:,:],
                                data_ref=self.era5_train,
                                config=self.config).load()
        print("compute spectra, evaluate, era5", era5.shape)
        esm = apply_transforms(self.esm_test[:,:,:],
                               data_ref=self.era5_train,
                               config=self.config).load()
        print("compute sprectra, evaluate", esm.shape)
        self.esm_psd = mean_rapsd(self.esm_test, normalize=True)
        self.era5_psd = mean_rapsd(self.era5_test, normalize=True)

    def compute_spectra_manually(self, esm_test_noised) -> None:
        """ Computes radially averaged power spectral densities of:
            - the ESM data
            - the noised ESM data
            - the ERA5 target data.
        """

        print("compute sprectra, evaluate", esm.shape)
        self.esm_noised_psd = mean_rapsd(esm_test_noised, normalize=True)
        self.esm_psd = mean_rapsd(self.esm_test, normalize=True)
        self.era5_psd = mean_rapsd(self.era5_test, normalize=True)

    def plot_spectra(self,
                     freq_min: Optional[float]=None,
                     psd_val: Optional[float]=None,
                     fname: Optional[str]=None) -> None:
        """ Plots the PSDs together with the intersection frequency. """

        self.log_diff = abs(np.log(self.era5_psd[0]) - np.log(self.esm_psd[0]))[1:]
        x_min = np.where(self.log_diff==self.log_diff.min())
        if freq_min is not None:
            self.freq_min = freq_min
        else:
            self.freq_min = self.era5_psd[1][1:][x_min]

        print(f"PSD intersection at freq={self.freq_min}")

        plt.figure(figsize=(7,5))

        mpl.rcParams['axes.linewidth'] = 1.5
        plt.tick_params(width=1.5)

        plt.subplot(1,1,1)

        plt.plot(self.era5_psd[1], self.era5_psd[0], label='MAR', c='k', lw=2)
        plt.plot(self.esm_noised_psd[1], self.esm_noised_psd[0], label='ESM noised', c='C4', lw=2)
        plt.plot(self.esm_psd[1], self.esm_psd[0], label='ESM',  c='orange', lw=2)

        plt.axvline(x=self.freq_min, c='tab:gray', ls='--', lw=2, label=r'$k^* = $'+f'{self.freq_min[0]:2.4f}')
        if psd_val is not None:
            plt.axhline(y=psd_val, c='tab:gray', ls='-', lw=2, label=r'$\mathrm{PSD}(k^*) = $'+f'{psd_val:2.1e}')

        plt.yscale("log", base=2)
        plt.xscale("log", base=2)
        plt.xlabel(r'Wavenumber')
        plt.ylabel('Power spectral density')
        plt.ylim(2**(-25), 2**(-1))
        plt.xlim(2**(-12), 2**(-1))
        plt.legend(frameon=False)

        #if fname is not None:
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        print(fname)
        #plt.show()





