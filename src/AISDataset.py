import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class AISDatasetMMSI(Dataset):
   def __init__(self, mmsi_file, data_dir, transform=None, target_transform=None):
      df = pd.read_csv(data_dir + mmsi_file)

      # will no longer be needed once i fix cleaning script
      df = df[df.VesselType.notna()]

      # cut out data that doesn't have enough data to train/test on
      val_counts = df.VesselType.value_counts()
      thresh = val_counts[val_counts > 1000]
      df = df[df.VesselType.isin(thresh.index)]

      # assign 0->C-1 labels to Vessel Type
      unique_type = df.VesselType.unique()
      zero_based = np.arange(len(unique_type))
      type_dict = dict(zip(unique_type,zero_based))
      df.VesselType = df.apply(lambda row: type_dict[row.VesselType],axis=1)

      # move frame to object's data
      self.df = df
      self.num_class = len(df.VesselType.unique())
      print("Created dataset with num_class=",self.num_class,", and length=",len(self.df))

   def __len__(self):
      return len(self.df)

   def __getitem__(self,idx):
      row = self.df.iloc[idx]
      
      # Only looking at Length and Width in uniqueMMSI.csv file.
      # Other files may have more inputs, but will have less data overall
      x = torch.Tensor([row["Length"],row["Width"],row["Draft"]])
      lbl = torch.Tensor([row["VesselType"]])
      return x,lbl
