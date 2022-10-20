import numpy as np


class DataProducer:
  # data = (x, y, dim_size)
  def __init__(self, data, num_steps, batch_size):
    num_ep = data.shape[0] #150
    num_frames = data.shape[1] #1000
    dim_size = data.shape[2] # 4

    self.data = data.reshape((-1, dim_size)) # flatten #150000
    self.num_steps = num_steps #40
    self.batch_size = batch_size #50

    num_vectors = num_ep * num_frames #150000
    batch_len = num_vectors // self.batch_size #3000
    self.epoch_size = (batch_len - 1) // self.num_steps # offset by 1 for teacher training #74

    self.data = self.data[0:batch_size*batch_len,:] #resize and removing remainder
    self.data = self.data.reshape((batch_size, batch_len, dim_size))

    self.counter = 0


  def reset(self):
    self.counter = 0

  def get_epoch_size(self):
    return self.epoch_size

  def next(self, ext_counter=None):
    if ext_counter == None:
      i = self.counter
      self.counter += 1
    else:
      i = ext_counter

    start = i * self.num_steps
    end = start + self.num_steps
    X = self.data[:, start: end, :]
    Y = self.data[:, (start+1): (end+1), :]

    return X, Y