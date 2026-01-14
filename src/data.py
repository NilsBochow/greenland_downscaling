from typing import List, Optional, Type, Union, Tuple

import torch 
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from dataclasses import dataclass
from dataclasses import replace
from matplotlib import pyplot as plt
from src.utils.transforms import apply_transforms

from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch
def _make_year_weights(years: np.ndarray,
                       boost_year: int = 2080,
                       boost: float = 10.0,
                       base: float = 1.0,
                       normalize: bool = True):
    """
    years: array of ints, len == len(dataset)
    Boost all samples with year >= boost_year by `boost`.
    """
    years = np.asarray(years)
    w = np.where(years >= boost_year, base * boost, base).astype(np.float32)
    if normalize:
        w *= (len(w) / w.sum())
    return torch.as_tensor(w, dtype=torch.float32)

def _make_summer_weights(months: np.ndarray,
                         summer_months=(6,7,8),
                         summer_boost=5.0,
                         base=1.0,
                         normalize=True):
    """
    months: array of ints in [1..12]
    summer_boost: multiplicative weight for summer samples
    """
    months = np.asarray(months)
    is_summer = np.isin(months, np.asarray(summer_months))
    w = np.where(is_summer, base * summer_boost, base).astype(np.float32)
    if normalize:
        w = w * (len(w) / w.sum())
    return torch.as_tensor(w, dtype=torch.float32)

def get_dataloaders(config, n_workers=1, use_mnist=False):
    if use_mnist:
        train_dataset = MNISTDataset(config, train=True)
        val_dataset   = MNISTDataset(config, train=False)
    else:
        train_dataset = GeoDataset("train", "ERA5", config)
        val_dataset   = GeoDataset("valid", "ERA5", config)

    # --- Summer oversampling for TRAIN only ---
    if getattr(config, "oversample_year_tail", True):
        # make sure your dataset exposes train_dataset.years (np array of ints)
        weights = _make_year_weights(
            getattr(train_dataset, "years"),
            boost_year=getattr(config, "boost_year", 2080),
            boost=getattr(config, "year_boost", 10.0),
            base=1.0,
            normalize=True
        )
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),   # epoch size ~ dataset size
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,                            # <-- key line
            shuffle=False,                              # don't shuffle when using a sampler
            num_workers=n_workers,
            pin_memory=True,
            persistent_workers=(n_workers > 0),
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=n_workers
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=n_workers
    )

    return {"train": train_loader, "val": val_loader}


class GeoDataset(torch.utils.data.Dataset):
    """ Dataset for ESM simulation and ERA5 reanalysis"""
    
    def __init__(self,
                 stage: str,
                 dataset_name: str,
                 config: dataclass,
                 epsilon: Optional[float]=0.0001,
                 transform_esm_with_target_reference: Optional[bool]=True):
        """ 
            stage: Train, valid or test.
            dataset_name: Either ESM or ERA5.
            config: Model configuration dataclass
            epsilon: Small constant for the log transform
            transform_esm_with_target_reference: Use target dataset to tranform the ESM data.
        """
        self.stage = stage
        self.config = config
        self.transforms = config.transforms
        self.epsilon = epsilon
        self.transform_esm_with_target_reference = transform_esm_with_target_reference

        self.target = None
        self.target_reference = None
        self.climate_model = None
        self.data = None
        self.insolation_map = None
        self.surface_map = None
        self.ts = None 
        self.target_ts = None
        self.target_reference_ts = None
        if config.lazy:
            self.cache = False
            self.chunks = {'time': 1}
        else:
            self.cache = True
            self.chunks = None

        assert(stage in ['train', 'valid', 'test', 'proj']), "stage needs to be train, valid or test"

        self.splits = {
                "train": [config.train_years],
                "valid": [config.valid_years],
                "test":  [config.test_years],
                "proj":  ['2015', '2050'],
        }

        print("splits", self.splits)
        self.pad = torch.nn.ZeroPad2d(config.pad_input)

        assert(dataset_name in ['ESM', 'ERA5']), f"Dataset name {dataset_name} not supported"

        if dataset_name == "ERA5":
            self.prepare_target_data()

        elif dataset_name == "ESM":
            self.prepare_climate_model_data()
        # Extract months from the time dimension
        self.months = self.data.time.dt.month.values  # Months as integers (1-12)
        self.insolation_map = self.load_insolation_maps()
        self.surface_map = self.load_surface_map()

        self.prepare_surface_temperature()
        self.years  = self.ts["time"].dt.year.values.astype(np.int16)

    def load_data(self, filename, is_reference=False):
        """ Loads data from file and applies some preprocessing.

        Args:
            is_reference: Loads data from the training period to be used as reference for transformations.
        """

        data_path: str = self.config.data_path + '/' + filename
        target = xr.open_dataset(data_path, use_cftime=True,
                               cache=self.cache, chunks=self.chunks)#["SMB"]
        target.astype('float32')

        #assert len(list(target.keys())) <= 1, "more than one variable detected in target dataset."
        self.config.predict_variable = list(target.keys())[0]
 
        target = target[self.config.predict_variable]

        if is_reference:
            target = target.sel(time=target.time.dt.year.isin(self.splits["train"]))
        else:
            target = target.sel(time=target.time.dt.year.isin(self.splits[self.stage]))

        if self.config.crop_data_latitude != (None,None):
            target = target.isel(latitude=slice(self.config.crop_data_latitude[0],
                                                self.config.crop_data_latitude[1]))

        if self.config.crop_data_longitude != (None,None):
            target = target.isel(longitude=slice(self.config.crop_data_longitude[0],
                                                 self.config.crop_data_longitude[1]))

        if self.config.use_float16:
            target = target.astype(np.float16)
            print("float 16")

        print(self.stage, target.shape, slice(self.splits[self.stage]))
        return target


    def load_surface_temperature(self, filename, is_reference=False): 
        data_path: str = self.config.data_path + '/' + filename
        target = xr.open_dataset(data_path, use_cftime=True,
                               cache=self.cache, chunks=self.chunks)["ST"]
        target.astype('float32')
        if is_reference:
            target = target.sel(time=target.time.dt.year.isin(self.splits["train"]))
        else:
            target = target.sel(time=target.time.dt.year.isin(self.splits[self.stage]))
        
        return target              

    def prepare_surface_temperature(self):
        """ Calls the target data loading and applies transformations.  """

        self.target_ts = self.load_surface_temperature(self.config.target_filename)
        self.target_reference_ts = self.load_surface_temperature(self.config.target_filename, is_reference=True)
        #self.num_samples_ts = len(self.target_ts.time.values)
        #modified_config = replace(self.config, transforms=['normalize_minus1_to_plus1'])

        self.ts = apply_transforms(self.target_ts, config = self.config, data_ref = self.target_reference_ts)

    def load_insolation_maps(self):
        data_path: str =  "data/datasets/insolation_masked.nc"
        target = xr.open_dataset(data_path, use_cftime=True,
                               cache=self.cache, chunks=self.chunks)["insolation"]
        target.astype('float32')
        #target = target / np.max(target)
        target = (target - np.min(target)) / (np.max(target) - np.min(target))
        target = target * 2 - 1
        #assert len(list(target.keys())) <= 1, "more than one variable detected in target dataset."

        return target

    def load_surface_map(self):
        data_path: str =  "data/datasets/SRF_masked.nc"
        target = xr.open_dataset(data_path, use_cftime=True,
                               cache=self.cache, chunks=self.chunks)["SRF"]
        target.astype('float32')
        target = (target - np.min(target)) / (np.max(target) - np.min(target))
        target = target * 2 - 1
        #assert len(list(target.keys())) <= 1, "more than one variable detected in target dataset."

        return target

    def prepare_climate_model_data(self):
        """ Calls the climate model data loading and applies transformations.  """

        self.climate_model = self.load_data(self.config.esm_filename)
        if self.transform_esm_with_target_reference:
            climate_model_reference = self.load_data(self.config.target_filename, is_reference=True)
        else:
            climate_model_reference = self.load_data(self.config.esm_filename, is_reference=True)
        self.num_samples = len(self.climate_model.time.values)
        print("num samples climate model data", self.num_samples)
        self.data = apply_transforms(self.climate_model, self.config, data_ref = climate_model_reference )
        print("applied transforms prepare climate data")

    def prepare_target_data(self):
        """ Calls the target data loading and applies transformations.  """

        self.target = self.load_data(self.config.target_filename)
        self.target_reference = self.load_data(self.config.target_filename, is_reference=True)
        self.num_samples = len(self.target.time.values)
        self.data = apply_transforms(self.target, self.config, data_ref = self.target_reference)
        


    def __getitem__(self, index):

        y = torch.from_numpy(self.data.isel(time=index).values).float().unsqueeze(0)
        y = self.pad(y)

        month = self.months[index] - 1  # Convert to 0-indexed month (0-11)
        #month = torch.tensor(month, dtype=torch.long)  # Convert to tensor
        ts = torch.from_numpy(self.ts.isel(time=index).values)#.float().unsqueeze(0)

        insolation_map = np.array(self.insolation_map[month, :, :])
        surface_map = np.array(self.surface_map)
 
        return y, month, surface_map, insolation_map, ts#, ts  # Return both the data and the month


    def __len__(self):
        return self.num_samples



