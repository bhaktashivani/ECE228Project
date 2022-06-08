import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
   def __init__(self,in_dim,out_dim,hidden_dim,num_layers,use_cuda):
      super(LSTM, self).__init__()
      
      self.layer_dim = num_layers
      self.hidden_dim = hidden_dim
      self.lstm = nn.LSTM(input_size=in_dim,hidden_size=hidden_dim,num_layers=num_layers)
      self.fc = nn.Linear(hidden_dim,out_dim)

      self.use_cuda = use_cuda
   
   def forward(self,x):
      h0 = torch.zeros(self.layer_dim,x.size(1),self.hidden_dim).requires_grad_()
      c0 = torch.zeros(self.layer_dim,x.size(1),self.hidden_dim).requires_grad_()
      if self.use_cuda:
         h0 = h0.cuda()
         c0 = c0.cuda()
      x,(hn,cn) = self.lstm(x,(h0.detach(),c0.detach()))
      x = torch.sigmoid(x[:,-1,:])
      x = self.fc(x)
      return x
