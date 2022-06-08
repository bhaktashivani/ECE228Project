import numpy as np
import torch
import torch.nn as nn

#from ellipsoidalNvector import LatLon

def evaluate(model,dataloader,batch_size,use_cuda=False):

   criterion = nn.MSELoss()

   running_loss = 0
   for batch, (inputs,labels) in enumerate(dataloader,0):
      #inputs = torch.tensor(inputs).float()
      
      labels = labels.squeeze_()
      labels = labels.type(torch.FloatTensor)
      #labels = torch.tensor(labels)
      
      if use_cuda:
         inputs = inputs.cuda()
   
      with torch.no_grad():
         output = model(inputs)

      if use_cuda:
         output = output.cpu()

      #outLatLon = LatLon(output(0),output(1))
      #lblLatLon = LatLon(labels(0),labels(1))

      running_loss += criterion(output,labels)
      
   return running_loss/batch_size
