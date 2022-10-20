'''
This module provides a helper function for reading data and create a Dataset for training the VAE
'''

import numpy as np
import os

def create_dataset(data_dir, type='obs'):
    
  filelist = os.listdir(data_dir)
  list_of_data = []

  for filename in filelist:
    raw_data = np.load(os.path.join(data_dir, filename))[type].reshape((-1, 64, 64, 3))
    list_of_data.append(raw_data)
  
  data = np.concatenate(list_of_data, axis=0)
  return Dataset(data)


class Dataset():
    def __init__(self, data):
        if data.shape[0] == 0:
            raise Exception("Dataset is empty")

        self.data = data
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, subscript):
        return self.data[subscript]