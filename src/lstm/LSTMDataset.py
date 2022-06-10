import os
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class LSTMDataset(Dataset):
   def __init__(self,data_dir,transform=None,target_transform=None):
      '''
      Rather than loading all data into memory, keep data in csv files and read as needed.
      To do this, an index is created upon dataset creation and then used to find
      the correct file and voyage ID when provided an index from the outside
      '''
      print("Loading dataset...")
      self.data_dir = data_dir
      self.file_list = os.listdir(data_dir)
      self.index_list = []
      total_samples = 0
      file_idx = 0
      print("num files = ",len(self.file_list))
      for f in self.file_list:
         df = pd.read_csv(data_dir + "/" + f)
         voyage_ids = df.VoyageNum.unique()
         for voyage_id in voyage_ids:
            voyage_df = df[df.VoyageNum == voyage_id]
            times = voyage_df.BaseDateTime
            # Times are seperated by 5 minute interval with minimum 25 min voyage time
            # expecting to take 4 samples and predict the 5th, therefore, use sliding
            # window of 4, then predict 4+1
            num_samples = len(times)-4
            sample_idxs = np.arange(total_samples,total_samples + num_samples)
            self.index_list.append([sample_idxs,file_idx,voyage_id])
            total_samples = total_samples + num_samples
         file_idx = file_idx+1
         if file_idx % 100 == 0:
            print("Loading for file ",file_idx)

      print("Dataset loaded with length ",total_samples)
      self.len = total_samples
            
   def __len__(self):
      return self.len

   def __getitem__(self,idx):
      x = torch.Tensor()
      y = torch.Tensor()
      for index in self.index_list:
         sample_idx = np.searchsorted(index[0],idx)
         if sample_idx < len(index[0]):
         #sample_idx = np.where(index[0] == idx)[0]
         #if sample_idx.size > 0:
         #   sample_idx = sample_idx[0]
            df = pd.read_csv(self.data_dir + "/" + self.file_list[index[1]])
            voyage_df = df[df.VoyageNum == index[2]]
            x = df.iloc[sample_idx:sample_idx+4]
            if (sample_idx+4) > len(df):
               print("idx: ",idx)
               print("file: ",self.file_list[index[1]])
               print("voyage_num: ", index[2])
               print("sample_idx: ",sample_idx)
               print("len(index[0]): ",len(index[0]))
               print(index[0])
            y = df.iloc[sample_idx+4]

            x = torch.Tensor(np.array([x.LAT.values,x.LON.values,x.SOG.values,x.COG.values,x.Heading.values]).T)
            y = torch.Tensor(np.array([y.LAT,y.LON]).T)
            return x,y
            
      return x,y      
      
