#!/bin/bash
# This script is meant to download a set amount of days/months/years 
# data from the site that holds the data. 
#
# WARNING: A single year's worth of data is about 100G, so be sure
#          that you have enough space before downloading too much.

site="https://coast.noaa.gov/htdata/CMSP/AISDataHandler/"
years=(2021)
months=$(seq -f "%02g" 12)
days=$(seq -f "%02g" 31)

for year in $years
do
   for month in $months
   do
      for day in $days
      do
         #echo $site$year"/AIS_"$year"_"$month"_"$day".zip"
         file=$site$year"/AIS_"$year"_"$month"_"$day".zip"
         wget $file
      done
   done
done
