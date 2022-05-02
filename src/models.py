import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerReLU(nn.Module):
   def __init__(self):
      super(TwoLayerReLU, self).__init__()
      
      # Input layer to increase as we can use more information from data file
      self.lin1 = nn.Linear(2,64)
   
      # Only using top 38 classes since they have over 100 data points
      self.lin2 = nn.Linear(64,38)

   def forward(self,x):
      x = self.lin1(x)
      x = F.relu(x)
      x = self.lin2(x)
      return F.log_softmax(x,dim=1)
