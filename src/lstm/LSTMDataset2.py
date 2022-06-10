import os
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class LSTMDataset2(Dataset):
   def __init__(self,data_dir,transform=None,target_transform=None):
      '''
      Load all data into single DataFrame
      '''

      print("Loading dataset...")
      self.data_dir = data_dir
      self.file_list = os.listdir(data_dir)
      self.index_list = []
      total_samples = 0
      file_idx = 0
      self.full_df = pd.DataFrame()
      print("num files = ",len(self.file_list))
      for f in self.file_list:
         df = pd.read_csv(data_dir + "/" + f)
         voyage_ids = df.VoyageNum.unique()
         
         for voyage_id in voyage_ids:
            voyage_df = df[df.VoyageNum == voyage_id]
            if (len(voyage_df) < 5):
               continue
            self.full_df = self.full_df.append(voyage_df.iloc[0:5])
            total_samples+=1
         file_idx = file_idx+1
         if file_idx % 100 == 0:
            print("Loading for file ",file_idx)

      print("Dataset loaded with length ",total_samples)
      self.len = total_samples
            
   def __len__(self):
      return self.len

   def __getitem__(self,idx):
      # input only first 20 min. Since 5 minute interval, that is only first 4 samples
      x = self.full_df.iloc[idx*5:idx*5+4]
      if (idx*5+4 > len(self.full_df)):
         print("idx: ",idx)
         print("num_samp: ", self.len)
         print("len(full_df): ",len(self.full_df))
         print("x: ",x)
      # 25th minute (5th sample) is the label
      y = self.full_df.iloc[idx*5+4]

      x = torch.Tensor(np.array([x.LAT.values,x.LON.values,x.SOG.values,x.COG.values,x.Heading.values]).T)
      # only predict [LAT,LON]
      y = torch.Tensor(np.array([y.LAT,y.LON]).T)
      return x,y
