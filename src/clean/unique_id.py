import pandas as pd
import numpy as np
import sys

def add_unique_id(input_file,output_file):
   '''
   Collect entries with valid Length, Width, and Draft entries for 3-D input to MLP
   
   Find all unique MMSI ID entries in the input file, choose a single entry for the ID, and 
   place in the output file. 
   
   Create the output file if does not exist yet
   '''
   df = pd.read_csv(input_file)
   df = df[df.VesselType.notna()]
   
   if 'TranscieverClass' in df.columns:
      df['TransceiverClass'] = df['TranscieverClass']

   cols = []
   if 'TransceiverClass' not in df.columns:
      cols = ['MMSI','VesselType','Length', 'Width', 'Draft', 'Cargo']
   else:
      cols = ['MMSI','VesselType','Length', 'Width', 'Draft', 'Cargo','TransceiverClass']
   df = df[cols]

   # Add additional columns here to keep for classification purposes
   df = df[(df.Length.notna()) & (df.Width.notna()) & (df.Draft.notna())]
   df_mmsi = df.MMSI.unique()

   uniq_df = pd.DataFrame(columns = cols);
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
