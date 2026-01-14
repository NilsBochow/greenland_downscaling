import xarray as xr
import numpy as np
from scipy.interpolate import griddata

# Load your low-resolution NetCDF file
input_file = "/p/projects/ou/labs/ai/Nils/arctic-downscaling/data/datasets/output_40km_ssp126_noresm.nc"
ds = xr.open_dataset(input_file)

# Original x and y coordinates (1D arrays of integer indices)
x_vals = ds.x.values
y_vals = ds.y.values

# Create a mesh grid for the original (x, y) coordinates
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

# Define the high-resolution target grid along the y and x axes
y_highres = np.linspace(y_vals.min(), y_vals.max(), ds.sizes['y'] * 8)  # Increase y resolution by a factor of 8
x_highres = np.linspace(x_vals.min(), x_vals.max(), ds.sizes['x'] * 8)  # Increase x resolution by a factor of 8

# Create a mesh grid for the high-resolution coordinates
x_grid, y_grid = np.meshgrid(x_highres, y_highres)

# Initialize a list to store the interpolated data for each time step
interpolated_smb = []

# Loop over each time step and interpolate
for t in range(ds.sizes['time']):
    # Extract the SMB data for the current time step
    smb_data = ds.SMB.isel(time=t).values
    
    # Flatten the x and y mesh grids and SMB data for griddata
    points = np.array((x_mesh.ravel(), y_mesh.ravel())).T
    values = smb_data.ravel()
    
    # Perform bilinear interpolation on the high-resolution grid
    smb_highres = griddata(points, values, (x_grid, y_grid), method="linear")
    
    # Append the interpolated data for this time step
    interpolated_smb.append(smb_highres)

# Convert the list of interpolated data to a NumPy array with dimensions (time, y, x)
interpolated_smb = np.stack(interpolated_smb, axis=0)

# Create a new high-resolution dataset
ds_highres = xr.Dataset(
    {"SMB": (("time", "y", "x"), interpolated_smb)},
    coords={"time": ds.time, "y": y_highres, "x": x_highres}
)

# Save the interpolated data to a new NetCDF file
output_file = "interpolated_5km.nc"
ds_highres.to_netcdf(output_file)
