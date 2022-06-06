import os
import sys
import pandas as pd

def collect_type(data_dir,type_num,output_dir):
   num_files_out = 0
   for f in os.listdir(data_dir):
      if f.endswith(".csv"):
         df = pd.read_csv(data_dir + "/" + f)
         df = df[df.VesselType == type_num]
         if (len(df) > 10):
            num_files_out = num_files_out + 1
            print("Num files:",num_files_out)
            df.to_csv(output_dir + "/" + f,index=False)
         

if __name__=='__main__':   
   data_dir = sys.argv[1]
   type_num = float(sys.argv[2])
   output_dir = sys.argv[3]
   collect_type(data_dir,type_num,output_dir)
