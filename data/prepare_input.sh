# script to prepare input for difusion model

model_name="NorESM2-ssp585-5km"
input_name="concatenated_SMB_NorESM2-ssp585-5km.nc"
variable="SMB"
echo $input_name


cdo setmisstoc,0 $input_name "temp_file_masked.nc"


# 2015 for future, 1950 for historical
cdo setcalendar,360_day -settaxis,2015-01-15,00:00:00,1month "data/datasets/temp_file_masked.nc" "data/datasets/temp_file_masked2.nc"
rm "data/datasets/temp_file_masked.nc"
cdo selindexbox,1,336,1,576 "data/datasets/temp_file_masked2.nc" "data/datasets/temp_file_masked3.nc"
rm "data/datasets/temp_file_masked2.nc"
ncrename -d lat,y -d lon,x  "data/datasets/temp_file_masked3.nc"
python mask_SMB.py --smb_file  "data/datasets/temp_file_masked3.nc" --model_name $model_name --variable $variable
rm "data/datasets/temp_file_masked3.nc"
