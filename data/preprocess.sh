#!/bin/bash

# Define a temporary output file
temp_output="temp_merged_SMB.nc"
temp_output2="temp_merged_SMB_2.nc"
output="output_name.nc"

# Check if there are any files matching the pattern
if ls concatenated_SMB*.nc 1> /dev/null 2>&1; then
    # Use cdo to concatenate all matching files into the temporary file
    cdo mergetime concatenated_SMB*.nc "$temp_output"

    echo "All files starting with 'concatenated_SMB' have been merged into $temp_output"
else
    echo "No files found with the pattern 'concatenated_SMB*.nc'"
fi

cdo setmisstoc,0 $temp_output $temp_output2
rm $temp_output
cdo setcalendar,360_day $temp_output2 $output
rm $temp_output2
rm $temp_output2 
