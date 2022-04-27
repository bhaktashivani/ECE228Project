#!/bin/bash
# This script is meant to download a set amount of days/months/years 
# data from the site that holds the data. 
#
# WARNING: A single year's worth of data is about 100G, so be sure
#          that you have enough space before downloading too much.

target_dir=/mnt/windows/Users/Public/Documents/ECE278/project/data/
site="https://coast.noaa.gov/htdata/CMSP/AISDataHandler/"
years=$(seq 2016 2020)
months=$(seq -f "%02g" 12)
days=$(seq -f "%02g" 31)

for year in $years
do
   for month in $months
   do
      for day in $days
      do
         file=$site$year"/AIS_"$year"_"$month"_"$day".zip"
         #echo $file
         wget $file -P $target_dir$year
      done
   done
done
