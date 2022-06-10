#!/bin/bash

# Loop through unzipped files and call python script to collect the unqiue MMSI IDs
# around the coast of California

data_dir=/path/to/unzipped/data
src_dir=/path/where/python/scripts
out_file="uniqueMMSI_SanDiego.csv"

#years=$(seq 2017 2021)
year=2021
months=$(seq -f "%02g" 12)
days=$(seq -f "%02g" 31)

for month in $months
do
   for day in $days
   do
      year_dir=$data_dir$year/
      name="Clean_AIS_"$year"_"$month"_"$day
      csv_file=$year_dir$name.csv

      python $src_dir\unique_id_SanDiego.py $csv_file $data_dir$out_file

   done
done
