import numpy as np
import pandas as pd
import sys

def clean_file(input_file, output_file):
   print("reading ", input_file)
   df = pd.read_csv(input_file)

   bbox = [-121,30,-116,35]
   mask = (df.LON > bbox[0]) & (df.LAT > bbox[1]) & (df.LON < bbox[2]) & (df.LAT < bbox[3])
   print("masking")
   df = df.loc[mask]

   drop_cols = ['VesselName','IMO','CallSign','Status',\
                'Length','Width','Draft','TranscieverClass']
   print("dropping cols")
   # !!! Need to deal with TranceiverClass vs TrancieverClass issue !!!!!! #
   df = df.drop(drop_cols,axis=1)
   print("dropping na")
   df = df[df.VesselType.notna()]
   print("writing csv")
   df.to_csv(output_file,index=False)

if __name__ == '__main__':
   input_file = sys.argv[1]
   output_file = sys.argv[2]
   clean_file(input_file, output_file)
