import pandas as pd
import numpy as np
import sys

def add_unique_id(input_file,output_file):
   print(input_file)
   df = pd.read_csv(input_file)

   df_mmsi = df.MMSI.unique()

   uniq_df = pd.DataFrame(columns=list(df.columns),dtype=object);
   try:
      uniq_df = pd.read_csv(output_file)
   except:
      print("First File, initiating output file")

   uniq_mmsi = uniq_df.MMSI.unique()      
   mmsi_list = list(set(df_mmsi) - set(uniq_mmsi))

   for mmsi in mmsi_list:
      uniq_df = uniq_df.append(df[df.MMSI == mmsi].iloc[0])

   uniq_df.to_csv(output_file,index=False)
   
if __name__ == '__main__':
   input_file = sys.argv[1]
   output_file = sys.argv[2]
   add_unique_id(input_file, output_file)
