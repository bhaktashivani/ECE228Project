#!/bin/bash
# Loop through AIS zip files, clean them, save cleaned data to another area

data_dir=/mnt/windows/Users/Public/Documents/ECE278/project/data/
clean_dir=/mnt/windows/Users/Public/Documents/ECE278/project/uniqueData/
src_dir=~/Documents/classes/graduate/physical/project/ECE228Project-/src/
out_file="uniqueMMSI_SanDiego.csv"

#years=$(seq 2017 2021)
years=$(2021)
months=$(seq -f "%02g" 12)
days=$(seq -f "%02g" 31)

for year in $years
do
   for month in $months
   do
      for day in $days
      do
         year_dir=$data_dir$year/
         name="AIS_"$year"_"$month"_"$day
         zip_file=$year_dir$name.zip
         csv_file=$year_dir$name.csv

         unzip $zip_file -d $year_dir
         python $src_dir\unique_id.py $csv_file $clean_dir$out_file

         rm $csv_file
         
      done
   done
done
