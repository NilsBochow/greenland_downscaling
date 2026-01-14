from typing import List, Optional, Type, Union, Tuple
import argparse
import xarray as xr
import numpy as np
from tqdm import tqdm
from xclim import sdba 
import pandas as pd
import src.utils.xarray_utils as xu
np.set_printoptions(threshold=np.inf)
from xclim.core.units import convert_units_to

class QuantileMapping():

    def __init__(self,
                 model_path: str,
                 target_path: str,
                 out_path: str,
                 num_quantiles: Optional[int]=200,
                 train_set: Optional[Tuple[str,str]]=['1950', '2014'],
                 test_set: Optional[Tuple[str,str]]=['1950', '2014'],
                 verbose: Optional[bool]=True
                 ):
        """Peforms quantile mapping on model data given a target dataset.

        Args:
            model_path: Path to simlulatin data in .nc format.
            target_path: Path to target data in .nc format.
            out_path: Path where to store the result.
            num_quantiles: Number of quantiles used for the mapping. 
            train_set: Beginnig and end of training period. 
            verbose: Enables verbose printing.
        """

        self.verbose = verbose

        self.model_path = model_path
        if self.verbose: print(model_path)
        
        self.target_path = target_path
        if self.verbose: print(target_path)

        self.out_path = out_path
        if self.verbose: print(out_path)
        
        self.train_set = train_set
        self.test_set = test_set
        self.num_quantiles = num_quantiles


    def load_data(self):
        """Loads the data from file. """
        print("test period", self.test_set)

        if self.verbose: print('loading data..')
        model = xr.open_dataset(self.model_path,
                                #chunks={'time': 50}) \
                                chunks=None) \
                                    .acabf \
                                    .astype(np.float32).load()
 
        self.model_historical = model.sel(time=slice(self.train_set[0], self.train_set[1])).convert_calendar('noleap',align_on='date')
        self.model_simulation = model.sel(time=slice(self.test_set[0], self.test_set[1])).convert_calendar('noleap', align_on='date')

        print(self.train_set[0], self.train_set[1])
        #if model.x[0] > 0:
        #    self.model_historical = xu.shift_longitudes(self.model_historical)
        #    self.model_simulation = xu.shift_longitudes(self.model_simulation)

        self.target = xr.open_dataset(self.target_path,
                                      #chunks={'time': 50} ) \
                                      chunks=None ) \
                                      .SMB \
                                      .astype(np.float32).load()

        self.target_historical = self.target.sel(time=slice(self.train_set[0], self.train_set[1])).convert_calendar('noleap',align_on='date')
        
        print("Target historical time length:", (self.target_historical.time.values))
        print("Model historical time length:", (self.model_historical.time.values))
        #self.target_historical = xu.remove_leap_year(self.target_historical)
        self.target_historical["time"] = self.model_historical.time

        if self.verbose: print('finished.')



    

    def run(self, method: str = "QDM"):
        """
        Perform bias correction on the whole grid (no Python loops).
        method: "EQM" (EmpiricalQuantileMapping) or "QDM" (QuantileDeltaMapping)
        """
        if self.verbose:
            print("fitting quantiles (vectorized)…")
        def _attach_units(da, default_units="kg m-2"):
            # Keep declared units if present; otherwise set a sensible default.
            if "units" not in da.attrs or not da.attrs["units"]:
                da = da.assign_attrs(units=default_units)
            return da
        def _force_same_units(*arrs, units="kg m-2"):
            fixed = []
            for da in arrs:
                da = da.copy()
                # kill any lingering encoded units that pint might read
                if hasattr(da, "encoding"):
                    da.encoding.pop("units", None)
                # overwrite attrs so xclim/pint think dims match
                da.attrs["units"] = units
                fixed.append(da)
            return fixed

        self.target_historical, self.model_historical, self.model_simulation = _force_same_units(
        self.target_historical, self.model_historical, self.model_simulation, units="kg m-2"
        )
        # 1) Ensure the property .units exists and reads from attrs
        #if not hasattr(xr.DataArray, "units"):
        #    xr.DataArray.units = property(lambda self: self.attrs.get("units", None))

       
        # 2) Attach units to all working arrays (pick ONE consistent unit!)
        self.target_historical = _attach_units(self.target_historical, "kg m-2")
        self.model_historical  = _attach_units(self.model_historical,  "kg m-2")
        self.model_simulation  = _attach_units(self.model_simulation,  "kg m-2")

        # Preserve seasonality: one mapping per calendar month
        group = sdba.Grouper("time.month")

        # Choose adjustment class
        if method.upper() == "QDM":
            AdjClass = sdba.adjustment.QuantileDeltaMapping
            # SMB can be negative -> additive deltas
            train_kwargs = dict(nquantiles=self.num_quantiles, group=group, kind="+")
        else:
            AdjClass = sdba.adjustment.EmpiricalQuantileMapping
            train_kwargs = dict(nquantiles=self.num_quantiles, group=group)

        # Train once over ALL (y, x). Works with dask chunks too.
        Adj = AdjClass.train(
            ref=self.target_historical,
            hist=self.model_historical,
            **train_kwargs
        )

        # Adjust the full future period at once
        mapped = Adj.adjust(sim=self.model_simulation)

        # Keep type small for IO; preserve coordinates/attrs
        self.result = mapped.astype("float32")

        if self.verbose:
            print("finished.")

    def save(self):
        """
        Save lazily with compression and consistent dimension order (time, y, x)
        so ncview can scroll through time easily.
        """
        # Wrap into a dataset
        ds_out = self.result.to_dataset(name="SMB")

        # Ensure coordinate order (time, y, x)
        expected_order = ["time", "y", "x"]
        for dim in expected_order:
            if dim not in ds_out.dims:
                raise ValueError(f"Missing expected dimension: {dim}")
        ds_out = ds_out.transpose("time", "y", "x")

        # Compression + chunking tuned for ncview
        enc = {
            "SMB": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "chunksizes": (
                    min(self.model_simulation.sizes.get("time", 120), 120),
                    256,
                    256,
                ),
                "_FillValue": -9.96921e36,
            }
        }

        # Write to NetCDF
        ds_out.to_netcdf(self.out_path, encoding=enc)

        if self.verbose:
            print(f"saved → {self.out_path}")
            print("Dimension order:", ds_out.SMB.dims)



def parse_command_line():
    """ Parses the command line options. """

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path",
                        help="Path to the model .nc file", type=str)

    parser.add_argument("-t", "--target_path",
                        help="Path to the target .nc file", type=str)

    parser.add_argument("-o", "--out_path",
                        help="Path to the output .nc file", type=str)

    parser.add_argument("-ts", "--training_start",
                        help="Start year of training data", type=int)

    parser.add_argument("-te", "--training_end",
                        help="Start year of training data", type=int)

    parser.add_argument("-nq", "--num_quantiles",
                        help="Number of quantiles", type=int)

    parser.add_argument("-v", "--verbose",
                        help="Verbose output", action='store_true')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_command_line()

    qm = QuantileMapping(model_path=args.model_path,
                         target_path=args.target_path,
                         out_path=args.out_path,
                         num_quantiles=args.num_quantiles,
                         train_set=[str(args.training_start), str(args.training_end)],
                         verbose=args.verbose)
    qm.load_data()
    qm.run()
    qm.save()