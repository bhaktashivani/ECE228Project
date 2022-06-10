import os
import sys
import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline

# Assuming files are organized by MMSI,
# seperate into different voyages and
# validate that a "good" voyage has occured, smooth it, then write out
# so that LSTM can use evenly-spaced time data

def smooth(df):
   '''
   Smooth Lat,Lon,COG,SOG,and Heading data over time using a Cubic Spline"
   '''
   df = df.drop_duplicates(subset='BaseDateTime',keep='first')
   times = np.array(df.BaseDateTime - df.BaseDateTime.iloc[0]) / np.timedelta64(5,'m')
   lat_spline = CubicSpline(times,df.LAT)
   lon_spline = CubicSpline(times,df.LON)
   sog_spline = CubicSpline(times,df.SOG)
   cog_spline = CubicSpline(times,df.COG)
   hdg_spline = CubicSpline(times,df.Heading)

   t_start = int(times[0])
   t_end = int(times[-1])
   t_range = np.arange(t_start,t_end,1)

   smooth_df = pd.DataFrame(columns = df.columns,dtype=object)
   smooth_df.BaseDateTime = t_range
   smooth_df.LAT = lat_spline(t_range)
   smooth_df.LON = lon_spline(t_range)
   smooth_df.SOG = sog_spline(t_range)
   smooth_df.COG = cog_spline(t_range)
   smooth_df.Heading = hdg_spline(t_range)
   smooth_df.MMSI = df.MMSI.iloc[0]
   smooth_df.VesselType = df.VesselType.iloc[0]
   smooth_df.VoyageNum = df.VoyageNum.iloc[0]
   return smooth_df

def split_into_voyages(file_name):
   '''
   Define voyages as longer than 10 minutes, and split if time breaks greater than 30min.
   Then, remove any voyages that didn't go for long enough or where the vessel did
   not move enough 
   '''
   voyage_time_split = 60 #minutes
   min_voyage_time = 25 #minutes
   df = pd.read_csv(file_name)
   df.BaseDateTime = pd.to_datetime(df.BaseDateTime)
   df = df.sort_values(by="BaseDateTime",ascending=True)
   df.reset_index(drop=True,inplace=True)

   front = df.BaseDateTime.iloc[:-1]
   front = pd.to_datetime(front)

   back = df.BaseDateTime.iloc[1:]
   back = pd.to_datetime(back)

   time_diff = back.to_numpy() - front.to_numpy()
   time_diff = time_diff / np.timedelta64(1,'m') #convert from nanoSec->sec->min

   voyage_flags = np.where(time_diff > voyage_time_split)[0] + 1 #marks new voyage
   
   new_df = pd.DataFrame()

   for i in range(len(voyage_flags)-1):
      start_idx = voyage_flags[i]
      end_idx = voyage_flags[i+1]
      not_enough_samples = end_idx - start_idx < 10

      voyage_time = df.BaseDateTime[end_idx] - df.BaseDateTime[start_idx]
      too_short = (voyage_time / np.timedelta64(1,'m')) < min_voyage_time
      if not_enough_samples or too_short:
         continue
      voyage_df = df.iloc[start_idx:end_idx]
      lats = voyage_df.LAT.to_numpy()
      lons = voyage_df.LON.to_numpy()
      not_enough_movement = np.sum(np.abs(lats[1:]-lats[:-1])) < 0.001 or np.sum(np.abs(lons[1:]-lons[:-1])) < 0.01
      if not_enough_movement:
         continue

      voyage_df["VoyageNum"] = i
      voyage_df = smooth(voyage_df)
      new_df = new_df.append(voyage_df)

   return new_df
      

def main(input_dir,output_dir):
   num_files = 0
   for f in os.listdir(input_dir):
      print("num_files",num_files)
      num_files = num_files + 1
      file_name = input_dir + "/"  + f
      df = split_into_voyages(file_name)
      if (len(df) > 10):
         df.to_csv(output_dir + "/" + f,index=False)

if __name__=='__main__':
   data_dir = sys.argv[1]
   output_dir = sys.argv[2]
   main(data_dir,output_dir)
