'''
This module allows you to convert data to encoded data.
'''

import numpy as np
import os
from data_loader import create_action_datalist, create_dataset
from visualize import encode_batch
from vae import ConvVAE
from dir_config import SERIES_DIR, WEIGHTS_DIR


def convert_to_series(vae, dataset, action_dataset):
  mu_dataset = []
  logvar_dataset = []
  for i in range(len(dataset)):
    data_batch = dataset[i]
    mu, logvar, z = encode_batch(vae, data_batch)
    mu_dataset.append(mu.astype(np.float16))
    logvar_dataset.append(logvar.astype(np.float16))

  action_dataset = np.array(action_dataset)
  mu_dataset = np.array(mu_dataset)
  logvar_dataset = np.array(logvar_dataset)

  np.savez_compressed(os.path.join(SERIES_DIR, f"series_{vae.z_size}.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)


if __name__ == "__main__":
  for z in [32]:  ## Put in the latent space you want to encode to
    Z_SIZE = z
    DATA_DIR = 'data'

    filename = f"vae_{Z_SIZE}.json"
    filepath = os.path.join(WEIGHTS_DIR, filename)
    vae = ConvVAE(z_size=Z_SIZE, batch_size=100, gpu_mode=True)
    vae.load_json(filepath)

    dataset = create_dataset(DATA_DIR)
    action_datalist = create_action_datalist(DATA_DIR)

    convert_to_series(vae, dataset, action_datalist)
