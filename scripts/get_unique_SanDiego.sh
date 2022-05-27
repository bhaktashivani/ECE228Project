#!/bin/bash
# Loop through AIS zip files, clean them, save cleaned data to another area

data_dir=/mnt/windows/Users/Public/Documents/ECE278/project/cleanData/unzipped/
clean_dir=/mnt/windows/Users/Public/Documents/ECE278/project/cleanData/unzipped/
src_dir=~/Documents/classes/graduate/physical/project/ECE228Project-/src/
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

      python $src_dir\unique_id_SanDiego.py $csv_file $clean_dir$out_file

   done
done
