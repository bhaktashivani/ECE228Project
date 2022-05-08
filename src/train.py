import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from AISDataset import AISDatasetMMSI
from models import TwoLayerReLU

le_file = "uniqueMMSI_withDraft.csv"
file_dir = "/mnt/windows/Users/Public/Documents/ECE278/project/uniqueData/"
dataset = AISDatasetMMSI(le_file,file_dir)

split = np.array([0.8,0.2]) # 80%, 20% split
data_len = len(dataset)
split_amount = split*data_len
split_amount = np.round(split_amount).astype(np.int)

train_set, test_set = torch.utils.data.random_split(dataset,split_amount.tolist())

model = TwoLayerReLU(dataset.num_class)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.02)

for epoch in range(5):
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
    running_loss = 0.0
    for i,(inputs,labels) in enumerate(train_loader,0):
        inputs = torch.tensor(inputs).float()
        
        # was running into errors with out these two lines. Stackoverflow told me to do this
        labels = labels.squeeze_()
        labels = labels.type(torch.LongTensor)

        labels = torch.tensor(labels)
        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        batch_size=len(labels)
        
    print("epoch:",epoch,". Loss:",running_loss/batch_size)
    running_loss = 0
