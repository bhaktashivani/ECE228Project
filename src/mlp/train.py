import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from AISDataset import AISDatasetMMSI
from models import TwoLayerReLU
from evaluate import evaluate

# !!! USER-CONFIG !!! #
le_file = "uniqueMMSI_withDraft.csv"
file_dir = "/mnt/windows/Users/Public/Documents/ECE278/project/uniqueData/"
#file_dir = "/path/where/le_file/exists"
dataset = AISDatasetMMSI(le_file,file_dir)

split = np.array([0.8,0.2]) # 80%, 20% split
data_len = len(dataset)
split_amount = split*data_len
split_amount = np.round(split_amount).astype(np.int)

train_set, test_set = torch.utils.data.random_split(dataset,split_amount.tolist())
test_loader = torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=True)

model = TwoLayerReLU(dataset.num_class)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=2e-3,betas=(0.3,0.99))

use_cuda = torch.cuda.is_available()
if use_cuda:
   model = model.cuda()
print("use_cuda: ",use_cuda)

test_acc = []
train_loss = []

for epoch in range(500):
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
    running_loss = 0.0
    for i,(inputs,labels) in enumerate(train_loader,0):
#        inputs = torch.tensor(inputs).float()
        
        # was running into errors with out these two lines. Stackoverflow told me to do this
        labels = labels.squeeze_()
        labels = labels.type(torch.LongTensor)
#        labels = torch.tensor(labels)

        if use_cuda:
            inputs,labels = inputs.cuda(),labels.cuda()

        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        batch_size=len(labels)
        
    eval_acc = evaluate(model,test_loader,use_cuda)
    epoch_loss = running_loss/batch_size
    test_acc.append(eval_acc)
    train_loss.append(epoch_loss)
    print("epoch:",epoch,". Loss:",epoch_loss,". Acc: ",eval_acc)

fig, axs = plt.subplots(2)
fig.suptitle('[10 Class] Training Loss(Top), and Test Accuracy(Bottom)')
axs[0].plot(train_loss)
axs[1].plot(test_acc)
plt.show()
