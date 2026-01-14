import numpy as np
import xarray as xr
import torch
from typing import Optional, Union
import math 

ArrayLike = Union[xr.DataArray, torch.Tensor]
def _coerce_s0_to_xarray(ref_s0, like: xr.DataArray) -> xr.DataArray:
    """
    Ensure ref_s0 is a 2-D (y,x) xarray.DataArray aligned to `like`'s spatial coords.
    Accepts torch.Tensor, numpy array, or xarray.DataArray with extra dims.
    """
    # pick a 2D "like" for coords (drop time if present)
    like2d = like.isel(time=0) if "time" in like.dims else like

    # get numpy array
    if isinstance(ref_s0, torch.Tensor):
        s = ref_s0.detach().cpu().numpy()
    elif isinstance(ref_s0, xr.DataArray):
        s = ref_s0.values
    else:
        s = np.asarray(ref_s0)

    # collapse to 2D [H, W]
    while s.ndim > 2:
        # prefer dropping singleton dims; otherwise reduce first non-singleton by median
        if s.shape[0] == 1:
            s = s[0]
        else:
            s = np.median(s, axis=0)
    if s.ndim != 2:
        raise ValueError(f"ref_s0 must be 2D after squeeze/reduction, got shape {s.shape}")

    # pick spatial dim names from `like`
    ydim = "y" if "y" in like2d.dims else like2d.dims[-2]
    xdim = "x" if "x" in like2d.dims else like2d.dims[-1]
    ycoord = like2d.coords[ydim] if ydim in like2d.coords else np.arange(s.shape[0])
    xcoord = like2d.coords[xdim] if xdim in like2d.coords else np.arange(s.shape[1])

    return xr.DataArray(s, dims=(ydim, xdim), coords={ydim: ycoord, xdim: xcoord}, name="s0")


def apply_transforms(
    data: ArrayLike,
    config,
    data_ref: Optional[xr.DataArray] = None,
    ref_mean: Optional[torch.Tensor]  = None,  # shape (y,x)
    ref_std:  Optional[torch.Tensor]  = None,  # shape (y,x)
    ref_min:  Optional[torch.Tensor]  = None,  # shape (y,x) or scalar
    ref_max:  Optional[torch.Tensor]  = None,  # shape (y,x)
    ref_s0:   Optional[torch.Tensor]  = None,
    ) -> ArrayLike:
    """
    Apply transforms to `data`, using either a full 3-D data_ref
    or the 2D stats (ref_mean, ref_std, ref_min, ref_max).
    """
    # --- figure out the log offset ---
    # if precomputed ref_min given, use that, else from data_ref
    if isinstance(data, torch.Tensor):
        x = data
        device = x.device
        
        # pre‐compute log offset if needed
        if 'log' in config.transforms:
            ref_min   = ref_min.min().item() - 1000
        if ('asinh' in config.transforms) and (ref_s0 is None):
            raise ValueError("Torch path needs ref_s0 (2D) for 'asinh'.")
        for tr in config.transforms:
            if tr == 'asinh':
                x = asinh_transform_torch(x, ref_s0)
            elif tr == 'log':
                data_ref_min = ref_min
                x = log_transform_torch(x, data_ref_min, config.epsilon)
            elif tr == 'standardize':
                x = standardize_torch(
                    x, mean2d=ref_mean, std2d=ref_std, epsilon=config.epsilon)
            elif tr == 'normalize':
                x = norm_torch(x, lo2d=ref_min, hi2d=ref_max, epsilon=config.epsilon)

            elif tr == 'normalize_minus1_to_plus1':
                x = norm_m1p1_torch(x, mean2d=ref_mean, std2d=ref_std, epsilon=config.epsilon)
            else:
                raise ValueError(f"Unknown transform: {tr}")

        return x
    else:
        if 'log' in config.transforms:
            if ref_min is not None:
                data_ref_min = float(ref_min.min()) - 1000  # collapse to scalar
            elif data_ref is not None:
                data_ref_min = float(data_ref.min()) - 1000
            else:
                raise ValueError("log requires data_ref or ref_min")

            data = log_transform(data, data_ref_min, config.epsilon)
            # also re-transform the reference if you still need it downstream
            if data_ref is not None:
                data_ref = log_transform(data_ref, data_ref_min, config.epsilon)
        if 'asinh' in config.transforms:
            if data_ref is None:
                raise ValueError("'asinh' with xarray path requires data_ref to compute per-pixel s0/mu/sig.")
            # compute z-space stats from TRAIN reference
            s0_da, mu_da, sig_da = fit_asinh_stats_xr(data_ref)
            data     = asinh_transform_xr(data, s0_da)
            data_ref = asinh_transform_xr(data_ref, s0_da)

        # --- STANDARDIZE ---
        if 'standardize' in config.transforms:
            # decide whether to use 3D or 2D stats
            if (ref_mean is not None and ref_std is not None):
                data = standardize(data, ref_mean=ref_mean, ref_std=ref_std,
                                epsilon=config.epsilon)
            elif data_ref is not None:
                data = standardize(data, data_ref=data_ref,
                                epsilon=config.epsilon)
                data_ref = standardize(data_ref, data_ref=data_ref,
                                epsilon=config.epsilon)
            else:
                raise ValueError("standardize needs data_ref or ref_mean/ref_std")

        if 'normalize' in config.transforms:
                if ref_min is not None and ref_max is not None:
                    data = norm_transform(data, lo=ref_min, hi=ref_max)
                elif data_ref is not None:
                    data = norm_transform(data, x_ref=data_ref)
                else:
                    raise ValueError("normalize needs data_ref or ref_min/ref_max")

        # --- NORMALIZE TO [-1,1] ---
        if 'normalize_minus1_to_plus1' in config.transforms:
            if ref_min is not None and ref_max is not None:
                data = norm_minus1_to_plus1_transform(data,
                                                    lo=ref_min, hi=ref_max,
                                                    epsilon=config.epsilon)
            elif data_ref is not None:
                data = norm_minus1_to_plus1_transform(data,
                                                    x_ref=data_ref,
                                                    epsilon=config.epsilon)
            else:
                raise ValueError("normalize_minus1_to_plus1 needs data_ref or ref_min/ref_max")

        return data



def apply_inverse_transforms(
    data: ArrayLike,
    config,
    *,
    # xarray inputs
    data_ref: Optional[xr.DataArray] = None,
    # torch inputs (2D stats)
    ref_mean: Optional[torch.Tensor] = None,
    ref_std:  Optional[torch.Tensor] = None,
    ref_min:  Optional[torch.Tensor] = None,
    ref_max:  Optional[torch.Tensor] = None,
    log_mean: Optional[torch.Tensor] = None,
    log_std: Optional[torch.Tensor] = None,
    ref_s0:   Optional[torch.Tensor] = None,  # <-- NEW
) -> ArrayLike:
    """
    Invert transforms in config.transforms on `data`, using either:
      - torch GPU path: data is a torch.Tensor and you provide ref_mean, ref_std, ref_min, ref_max (all 2D Tensors)
      - xarray CPU path: data is xr.DataArray and you provide data_ref (3D) or 2D DataArrays
    """
    # --- torch GPU path ---
    if isinstance(data, torch.Tensor):
        x = data
        
        # reverse transforms
        for tr in reversed(config.transforms):
            #print("tr", tr)
            if tr == 'log':
                if data_ref_min is None:
                    raise ValueError("Torch log inversion needs ref_min")
                x = inv_log_torch(x, data_ref_min, config.epsilon)
            elif tr == 'standardize':
                x = inv_standardize_torch(x, ref_mean, ref_std, config.epsilon) #need log mean and std if log is in transforms
            elif tr == 'asinh':
                if ref_s0 is None:
                    raise ValueError("'asinh' inversion needs either ref_s0 or data_ref to compute it")
                x = inv_asinh_transform_torch(x, ref_s0)

            elif tr == 'normalize_minus1_to_plus1':
                x = inv_norm_m1p1_torch(x, ref_min, ref_max, config.epsilon)
            elif tr == 'normalize':
                x = inv_norm_torch(x, ref_min, ref_max)
            else:
                raise ValueError(f"Unknown transform: {tr}")
        return x

    # --- xarray CPU path ---
    # ensure xarray DataArrays
    else:

        # prepare log offset
        if 'log' in config.transforms:
            if ref_min is not None:
                data_ref_min = float(ref_min.min()) - 1000
            elif data_ref is not None:
                data_ref_min = data_ref.min() - 1000
            else:
                raise ValueError("log inversion needs data_ref or ref_min")
            #print("data_ref_min normal", data_ref_min)
        # invert in reverse order, recomputing data_ref per-step
        if 'asinh' in config.transforms:
            if data_ref is None:
                raise ValueError("'asinh' with xarray path requires data_ref to compute per-pixel s0/mu/sig.")
            # compute z-space stats from TRAIN reference
            if ref_s0 is not None and not isinstance(ref_s0, xr.DataArray):
                    ref_s0 = _coerce_s0_to_xarray(ref_s0, like=data if "y" in data.dims else data.isel(time=0))
            #s0_da, mu_da, sig_da = fit_asinh_stats_xr(data_ref)
            if isinstance(data, xr.DataArray) and isinstance(ref_s0, torch.Tensor):
                ref_s0 = xr.DataArray(
                    ref_s0.detach().cpu().numpy(),
                    dims=("y","x"),
                    coords={"y": data.y, "x": data.x},
                    name="s0",
                )

          
        for i, tr in enumerate(reversed(config.transforms)):
            
            orig_idx = len(config.transforms) - 1 - i
            # recompute data_ref up to this step
            if data_ref is None:
                current_ref = None
            else:
                current_ref = data_ref.copy()
                for t in config.transforms[:orig_idx]:
                    if t == 'log':
                        current_ref = log_transform(current_ref, data_ref_min, config.epsilon)
                    elif t == 'asinh':
                        current_ref = asinh_transform_xr(current_ref, ref_s0)
    
                    elif t == 'standardize':
                        current_ref = standardize(current_ref, current_ref, config.epsilon)
                    elif t == 'normalize_minus1_to_plus1':
                        current_ref = norm_minus1_to_plus1_transform(current_ref, current_ref, config.epsilon)
                    elif t == 'normalize':
                        current_ref = norm_transform(current_ref, current_ref)
            # apply inverse
            if tr == 'log':
                data = inv_log_transform(data, data_ref_min, config.epsilon)
            elif tr == 'standardize':
                if current_ref is None:
                    raise ValueError("standardize inversion needs data_ref or ref_mean/std2d")
                data = inv_standardize(data, current_ref, config.epsilon)
            elif tr == 'asinh':
                data = inv_asinh_transform_xr(data, ref_s0)
            elif tr == 'normalize_minus1_to_plus1':
                if current_ref is None:
                    raise ValueError("norm m1p1 inversion needs data_ref")
                data = inv_norm_minus1_to_plus1_transform(data, current_ref, config.epsilon)
            elif tr == 'normalize':
                if current_ref is None:
                    raise ValueError("normalize inversion needs data_ref")
                data = inv_norm_transform(data, current_ref)
            else:
                raise ValueError(f"Unknown transform: {tr}")
        return data


def log_transform(x, data_ref_min, epsilon):
    offset = max(epsilon, epsilon - data_ref_min)
    return np.log(x + offset) - np.log(offset) 



def inv_log_transform(x, data_ref_min, epsilon):
    offset = max(epsilon, epsilon - data_ref_min)
    return np.exp(x + np.log(offset)) - (offset)


def standardize(x: xr.DataArray,
                data_ref: xr.DataArray = None,
                ref_mean: xr.DataArray = None,
                ref_std: xr.DataArray  = None,
                epsilon: float = 0.0001) -> xr.DataArray:
    """
    Standardize `x` using either:
      - data_ref (3D: time,y,x) → compute mean/std over time
      - or directly provided ref_mean/ref_std (2D: y,x)
    """
    # decide where to get mean/std

    if isinstance(x, torch.Tensor):
        if (ref_mean is None) or (ref_std is None):
            raise ValueError("When x is a Tensor you must pass mean2d/std2d as Tensors")
        return (x - ref_mean) / (ref_std + epsilon)
    else:
        if data_ref is not None:
            mean = data_ref.mean(dim='time')
            std = data_ref.std(dim='time')
        elif (ref_mean is not None) and (ref_std is not None):
            mean = ref_mean
            std = ref_std
        else:
            raise ValueError("Must provide either data_ref or both ref_mean and ref_std")

        return (x - mean) / (std + epsilon)


def inv_standardize(x: xr.DataArray,
                    data_ref: xr.DataArray = None,
                    ref_mean: xr.DataArray  = None,
                    ref_std: xr.DataArray   = None,
                    epsilon: float = 0.0001) -> xr.DataArray:
    """
    Invert standardization on `x` using either:
      - data_ref (3D: time,y,x) → compute mean/std over time
      - or directly provided ref_mean/ref_std (2D: y,x)
    """
    if data_ref is not None:
        mean = data_ref.mean(dim='time')
        std = data_ref.std(dim='time')
    elif (ref_mean is not None) and (ref_std is not None):
        mean = ref_mean
        std = ref_std
    else:
        raise ValueError("Must provide either data_ref or both ref_mean and ref_std")
    
    return x * (std + epsilon) + mean


def norm_transform(x, x_ref):
    return (x - x_ref.min(dim='time'))/(x_ref.max(dim='time') - x_ref.min(dim='time'))


def inv_norm_transform(x, x_ref):
    return x * (x_ref.max(dim='time') - x_ref.min(dim='time')) + x_ref.min(dim='time')


def norm_minus1_to_plus1_transform(x, x_ref, epsilon, use_quantiles=False, q_max=0.999):
    if use_quantiles: 
        x = (x - x_ref.quantile(1-q_max,dim='time'))/(x_ref.quantile(q_max,dim='time') - x_ref.quantile(1-q_max,dim='time'))
    else:
        x = (x - x_ref.min())/(x_ref.max() - x_ref.min() + epsilon)
    x = x*2 - 1
    return x 


def inv_norm_minus1_to_plus1_transform(x, x_ref,epsilon, use_quantiles=False, q_max=0.999):
    x = (x + 1)/2
    if use_quantiles: 
        x = x * (x_ref.quantile(q_max, dim='time') - x_ref.quantile(1-q_max,dim='time')) + x_ref.quantile(1-q_max, dim='time')
    else:
        x = x * (x_ref.max() - x_ref.min() + epsilon) + x_ref.min()
    return x


def log_transform_torch(x: torch.Tensor, data_ref_min, epsilon) -> torch.Tensor:
    offset = max(epsilon, epsilon - data_ref_min)
    return torch.log(x + offset) - math.log(offset)

def standardize_torch(x: torch.Tensor,
                      mean2d: torch.Tensor,
                      std2d:  torch.Tensor,
                      epsilon: float = 0.0001) -> torch.Tensor:
    # mean2d/std2d: shape (y, x)
    return (x - mean2d) / (std2d + epsilon)

def norm_torch(x: torch.Tensor,
               lo2d: torch.Tensor,
               hi2d: torch.Tensor,
               epsilon: float = 0.0001) -> torch.Tensor:
    # simple min–max 0→1
    return (x - lo2d) / (hi2d - lo2d + epsilon)

def norm_m1p1_torch(x: torch.Tensor,
                    lo2d: torch.Tensor,
                    hi2d: torch.Tensor,
                    epsilon: float = 0.0001) -> torch.Tensor:
    # min–max to [−1,1]
    x01 = (x - lo2d) / (hi2d - lo2d + epsilon)
    return x01 * 2 - 1

def inv_log_torch(x: torch.Tensor, data_ref_min, epsilon) -> torch.Tensor:
    offset = max(epsilon, epsilon - data_ref_min)
    return torch.exp(x + math.log(offset)) - offset

def inv_standardize_torch(x: torch.Tensor,
                          mean2d: torch.Tensor,
                          std2d:  torch.Tensor,
                          epsilon: float = 0.0001) -> torch.Tensor:
    return x * (std2d + epsilon) + mean2d

def inv_norm_torch(x: torch.Tensor,
                   lo2d: torch.Tensor,
                   hi2d: torch.Tensor) -> torch.Tensor:
    return x * (hi2d - lo2d) + lo2d

def inv_norm_m1p1_torch(x: torch.Tensor,
                        lo2d: torch.Tensor,
                        hi2d: torch.Tensor,
                        epsilon: float = 0.0001) -> torch.Tensor:
    x = (x + 1) / 2
    return x * (hi2d - lo2d + epsilon) + lo2d
def fit_asinh_stats_xr(y_train: xr.DataArray, floor_s0: float = 1.0e-3):
    """
    Robust per-pixel stats for asinh scaling using TRAIN years.
    y_train dims: (time, y, x) (or (..., time, y, x); 'time' must exist)
    Returns DataArrays: s0, mu, sig with dims (y,x).
    """
    abs_y = xr.apply_ufunc(np.abs, y_train)
    q25 = abs_y.quantile(0.25, dim="time")
    q75 = abs_y.quantile(0.75, dim="time")
    s0  = (q75 - q25) / 1.349
    s0  = xr.where(s0 < floor_s0, floor_s0, s0)

    z = xr.apply_ufunc(np.arcsinh, y_train / s0)
    mu  = z.median(dim="time")
    sig = (z.quantile(0.75, dim="time") - z.quantile(0.25, dim="time")) / 1.349
    sig = xr.where(sig < 1e-3, 1e-3, sig)
    return s0, mu, sig

def asinh_transform_xr(x: xr.DataArray, s0: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.arcsinh, x / s0)

def inv_asinh_transform_xr(z: xr.DataArray, s0: xr.DataArray) -> xr.DataArray:
    return s0 * xr.apply_ufunc(np.sinh, z)

# ---------- asinh helpers (torch) ----------
def asinh_transform_torch(x: torch.Tensor, s0_2d: torch.Tensor) -> torch.Tensor:
    # x: [B,1,H,W] or [B,C,H,W]; s0_2d: [H,W] (broadcasted per-batch)
    return torch.asinh(x / s0_2d)

def inv_asinh_transform_torch(z: torch.Tensor, s0_2d: torch.Tensor) -> torch.Tensor:
    return s0_2d * torch.sinh(z)
