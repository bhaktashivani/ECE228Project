import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerReLU(nn.Module):
   def __init__(self,num_class=10):
      super(TwoLayerReLU, self).__init__()
      
      # Input layer to increase as we can use more information from data file
      self.lin1 = nn.Linear(3,32)

      self.batch_norm = nn.BatchNorm1d(32)
   
      # Only using top 10 classes since they have over 1000 data points
      self.lin2 = nn.Linear(32,num_class)

   def forward(self,x):
      x = self.lin1(x)
      x = self.batch_norm(x)\

      x = torch.relu(x)
   
      # Final layer output
      x = self.lin2(x)
      return F.log_softmax(x,dim=1)

class ThreeLayerSigmoid(nn.Module):
   def __init__(self,num_class=10):
      super(TwoLayerReLU, self).__init__()
      
      # Input layer to increase as we can use more information from data file
      self.lin1 = nn.Linear(3,32)
      self.linMid = nn.Linear(32,32)

      self.batch_norm = nn.BatchNorm1d(32)
   
      # Only using top 10 classes since they have over 1000 data points
      self.lin2 = nn.Linear(32,num_class)

   def forward(self,x):
      x = self.lin1(x)
      x = self.batch_norm(x)\

      # Switch sigmoid and ReLU to see which has better functionality
      x = torch.sigmoid(x)
   
      # Additional middle layer to see if better results achieved 
      x = self.linMid(x)
      x = self.batch_norm(x)
      x = torch.sigmoid(x)

      # Final layer output
      x = self.lin2(x)
      return F.log_softmax(x,dim=1)
