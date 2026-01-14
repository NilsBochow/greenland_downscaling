# GrIS Consistency downscaling

Consistency-model-based downscaling for Greenland Ice Sheet surface mass balance (SMB) and surface temperature (ST). This code is accompanying the paper **Physics-constrained generative machine learning-based high-resolution downscaling of Greenland's surface mass balance and surface temperature**. Pre-print available [here](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-3927/).
The code trains either a score-based SDE model or a consistency model, conditioned on auxiliary maps (insolation and surface) and optionally temperature fields, then generates high-resolution samples saved as NetCDF.
Training, test data and other files will be made available on [Zenodo](doi.org/10.5281/zenodo.18241574).


## What’s in this repo
- `main.py`: training entry point; wires config, model selection, and training loop.
- `src/configuration.py`: config dataclasses and CLI overrides.
- `src/data.py`: dataset and dataloader logic (xarray + PyTorch), including oversampling of late years.
- `src/sde_model/`: score-based diffusion model (training + inference) NOTE: this code is not used in the accompanying paper.
- `src/consistency_model/`: consistency model (training + inference).
- `sample_consistency.py`: sampling script for the consistency model.
- `sample_bridge.py`: constrained/bridge sampling (initial conditions from an ESM). 
- `quantile_mapping.py`: quantile mapping utility.
- `coarsen_MAR_linear.py`, `coarsening_MAR_adaptive.py`: coarsening and interpolation scripts.
- `data/`: example datasets, stats, and helper scripts.

## Requirements
This repo ships a full conda environment in `environment.yml`, including CUDA, PyTorch, and NetCDF tooling.

Create the environment:
```bash
conda env create -f environment.yml
```

Activate it (as defined in the file):
```bash
conda consistency_smb
```


## Quick setup
If you just want to downscale your own fields without retraining and changing code, you just have to change the file paths in `sample_bridge.py`. 
Note: Your files need to have a dimension of 336x576 and need the same grid/projection as ISMIP6 (`data/ISMIP6_Extensions_05000m_grid.nc`). 
Remapping to the ISMIP6 grid is, for example, possible with `ncremap`: `ncremap -i input.nc -o output.nc -d ISMIP6_Extensions_05000m_grid.nc`.

## Data layout and paths
The code expects NetCDF inputs on disk and paths have to be adjusted for your use-case.

If you are running locally, update these paths before training/sampling:
- `src/configuration.py`: set `DataConfig.out_path`, `target_filename`, and any dataset names you use.
- `sample_consistency.py` and `sample_bridge.py`: update `/data/datasets/...` and `/data/results_oversample_asinh` paths.
- `src/consistency_model/inference.py`: update `transform_stats_*` paths.

## Training
`main.py` chooses the model from `--diffusion_model` (`consistency` or `ve`). Training uses PyTorch Lightning and expects a GPU.

Example:
```bash
python main.py --diffusion_model=consistency --n_epochs=200 --batch_size=4
```

Key training behaviors:
- Random year splits are created in `TrainingConfig` on import (see `src/configuration.py`).
- Oversampling of late years is enabled by default (`oversample_year_tail = True`).
- Training checkpoints are saved to `/data/results_oversample_asinh/` in `src/training.py` (hardcoded).

## Sampling
- **Consistency model sampling** (unconditional):
  ```bash
  python sample_consistency.py --diffusion_model=consistency --batch_size=1
  ```
  Writes a NetCDF output (currently `/sample_consistency.nc`).

- **Bridge / constrained sampling** (uses ESM initial conditions):
  ```bash
  python sample_bridge.py --diffusion_model=consistency --batch_size=1
  ```
  Writes NetCDF outputs to `/data/datasets/save_*.nc`.

Both scripts assume precomputed transform stats and training targets are available on disk. Update paths as needed.

## Data preprocessing utilities
- `data/preprocess.sh`: merges monthly SMB files, fixes missing values, and sets a 360-day calendar.
- `data/prepare_input.sh`: prepares ESM inputs (masking, time axis, cropping, renaming).
- `coarsen_MAR_linear.py`: linear interpolation + nearest-neighbor fill for coarsened data.
- `coarsening_MAR_adaptive.py`: adaptive average pooling to create coarsened fields.
- `mask_SMB.py`: masks input field to ice mask of ISMIP6.

## Configuration reference
The CLI overrides in `src/configuration.py` include (not exhaustive):
- `--diffusion_model` (`consistency` or `ve`)
- `--n_epochs`, `--batch_size`, `--lr`
- `--network_resolution`, `--channels`, `--down_block_types`, `--up_block_types`
- `--sigma_min`, `--sigma_max`
- `--target_filename`, `--esm_filename`, crop options, and precision flags

## Notes
- The code assumes a GPU (`accelerator='gpu'` in `src/training.py`).
- Several scripts save plots and outputs to absolute paths. Search for `"/data/"` and adjust if needed.

## Project structure
```
.
├── main.py
├── run.sh
├── sample_consistency.py
├── sample_bridge.py
├── quantile_mapping.py
├── coarsen_MAR_linear.py
├── coarsening_MAR_adaptive.py
├── data/
└── src/
    ├── configuration.py
    ├── data.py
    ├── training.py
    ├── consistency_model/
    ├── sde_model/
    └── utils/
```
