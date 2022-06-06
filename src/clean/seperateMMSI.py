import os
import sys

import pandas as pd

# Get all time series data for a specific MMSI that was found using 
# previous uniqueMMSI script

def split(clean_df,mmsi_list,output_dir):
   
   num_files_created = 0
   for mmsi in mmsi_list:
      mmsi_df = pd.DataFrame()
      mmsi_file = output_dir + "/" + str(mmsi) + ".csv"
      new_file = False
      try:
         mmsi_df = pd.read_csv(mmsi_file)
      except:
         new_file = True

      mmsi_data = clean_df[clean_df.MMSI == mmsi]
      #clean_df = clean_df.drop(mmsi_data.index)
      if (len(mmsi_data) > 0):
         if (new_file):
            num_files_created = num_files_created+1
         mmsi_df = mmsi_df.append(mmsi_data)
         mmsi_df.to_csv(mmsi_file,index=False)
   
   return num_files_created

def separate(input_dir,output_dir):
   uniqueMMSI_df = pd.read_csv(input_dir + "/uniqueMMSI_SanDiego.csv")
   unique_mmsi_list = list(uniqueMMSI_df.MMSI.unique())
   clean_file_list = []
   
   for f in os.listdir(input_dir): 
      if f.startswith("Clean_AIS_"): 
         clean_file_list.append(input_dir + '/' + f)
          
   for clean_file in clean_file_list:
      print(clean_file)
      clean_df = pd.read_csv(clean_file)
      num_files_created = split(clean_df,unique_mmsi_list,output_dir)
      print("Num files created: ", num_files_created)
   

if __name__== '__main__':
   input_dir = sys.argv[1]
   output_dir = sys.argv[2]
   separate(input_dir,output_dir)
