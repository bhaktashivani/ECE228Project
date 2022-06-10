#!/bin/bash
# Loop through AIS zip files, clean them, save cleaned data to another area

data_dir=/path/to/raw/data
clean_dir=/path/for/clean/data
src_dir=/path/where/python/scripts/live
years=$(seq 2017 2021)
months=$(seq -f "%02g" 12)
days=$(seq -f "%02g" 31)

for year in $years
do
   mkdir -p $clean_dir$year
   for month in $months
   do
      for day in $days
      do
         year_dir=$data_dir$year/
         name="AIS_"$year"_"$month"_"$day
         zip_file=$year_dir$name.zip
         csv_file=$year_dir$name.csv

         clean_year_dir=$clean_dir$year/
         clean_name="Clean_"$name
         clean_zip=$clean_year_dir$clean_name.zip
         clean_csv=$clean_year_dir$clean_name.csv
         
         unzip $zip_file -d $year_dir
         python $src_dir\clean_data.py $csv_file $clean_csv
         zip -r $clean_zip $clean_csv

         rm $csv_file
         rm $clean_csv
         
      done
   done
done
