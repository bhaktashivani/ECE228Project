import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from LSTMDataset2 import LSTMDataset2
from LSTMDataset import LSTMDataset
from models import LSTM
from evaluate import evaluate

file_dir = "/mnt/windows/Users/Public/Documents/ECE278/project/cleanData/type37_2021_forLSTM"
dataset = LSTMDataset2(file_dir)
#dataset = LSTMDataset(file_dir)

split = np.array([0.8,0.2]) # 80%, 20% split
data_len = len(dataset)
split_amount = split*data_len
split_amount = np.round(split_amount).astype(np.int32)

batch_size = 1024
train_set, test_set = torch.utils.data.random_split(dataset,split_amount.tolist())
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)

use_cuda = torch.cuda.is_available()
model = LSTM(5,2,128,2,use_cuda)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01,betas=(0.9,0.99))
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=0.95)

if use_cuda:
   model = model.cuda()
print("use_cuda: ",use_cuda)

test_loss = []
train_loss = []

for epoch in range(50):
   train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
   running_loss = 0.0
   for i,(inputs,labels) in enumerate(train_loader,0):
        
#      print("idx: ",i)     
        # was running into errors with out these two lines. Stackoverflow told me to do this
      labels = labels.squeeze_()
      labels = labels.type(torch.FloatTensor)

      if use_cuda:
          inputs,labels = inputs.cuda(),labels.cuda()

      optimizer.zero_grad()
      output = model(inputs)
      
      loss = criterion(output,labels)
      loss.backward()
      optimizer.step()
        
      batch_loss = loss.item()
    #  print("epoch:",epoch,". Running Loss:",batch_loss / batch_size)

      running_loss += batch_loss

      #print("evaluating")
      #eval_loss = evaluate(model,test_loader,use_cuda)
      #epoch_loss = running_loss/batch_size
      #test_loss.append(eval_loss)
      #train_loss.append(epoch_loss)
      #print("epoch:",epoch,". Loss:",epoch_loss,". Eval: ",eval_loss)
      

#   scheduler.step() 

   eval_loss = evaluate(model,test_loader,batch_size,use_cuda)
   epoch_loss = running_loss/batch_size
   test_loss.append(eval_loss)
   train_loss.append(epoch_loss)
   print("epoch:",epoch,". Loss:",epoch_loss,". Eval: ",eval_loss)
#
fig, axs = plt.subplots(2)
fig.suptitle('Training Loss(Top), and Test Loss(Bottom)')
axs[0].plot(train_loss)
axs[0].set_ylabel("Training Loss")
axs[1].plot(test_loss)
axs[1].set_ylabel("Test Loss")
axs[1].set_xlabel("Epoch")
plt.show()
