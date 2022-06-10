#!/bin/bash
# Loop through AIS zip files, clean them, save cleaned data to another area

data_dir=/path/to/raw/data
clean_dir=/path/for/clean/data
src_dir=/path/where/python/scripts/live
out_file="uniqueMMSI_withDraft.csv"

years=$(seq 2017 2021)
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
