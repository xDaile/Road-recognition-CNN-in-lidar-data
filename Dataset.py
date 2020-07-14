#!/usr/bin/env python3

import torch

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, tensorDict,GTDict):
        'Initialization'
        self.list_IDs = list_IDs
        self.tensorDict=tensorDict
        self.GTDict=GTDict
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)


  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        key = self.list_IDs[index]

        # Load data and get label
        X = torch.load(self.tensorDict[key])#HERE I ENDED
        y = torch.load(self.GTDict[key])
        X=X.to(device)
        y=y.to(device)
        return X, y,key
