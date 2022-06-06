import numpy as np
import torch

def evaluate(model,dataloader,use_cuda=False):
   y_pred=[]
   y = []

   for batch, (inputs,labels) in enumerate(dataloader,0):
      #inputs = torch.tensor(inputs).float()
      
      labels = labels.squeeze_()
      labels = labels.type(torch.LongTensor)
      #labels = torch.tensor(labels)
      
      if use_cuda:
         inputs = inputs.cuda()
   
      with torch.no_grad():
         output = model(inputs)

      y.append(labels)
      preds = torch.argmax(output,axis=1).cpu()
      if use_cuda:
         preds = preds.cpu()
      y_pred.append(preds)

      
   y_pred = np.hstack(y_pred)
   y = np.hstack(y)
   acc = np.mean(y_pred==y)

   return acc*100
