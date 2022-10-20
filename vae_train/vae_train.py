'''
This module defines a method for training a VAE.
'''

import os
import sys
import tensorflow as tf
import numpy as np

### Configure if you want to use weighted or unweighted vae ###
#from vae import ConvVAE, reset_graph, Weights
from vae_weights import ConvVAE, reset_graph, Weights

from data_loader import create_dataset
from visualize import visualize

os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
DATA_DIR = "data"

def _print_header(num_epoch, z_size, stream):
  # train loop:
  print(f"Num of epochs: {num_epoch}", file=stream)
  print(f"Z size: {z_size}", file=stream)
  print(Weights, file=stream)
  print("----------------------------------------------------------", file=stream)

  header_titles = ["step", "loss", "recon_loss", "kl_loss"] + list(Weights.keys())

  print(','.join(header_titles), file=stream)

def _print_stats(stats, stream):

  train_loss, r_loss, kl_loss, train_step, _, other_losses = stats
  vals_to_print = [train_step+1, train_loss, r_loss, kl_loss] + list(other_losses.values())
  to_print = [str(elem) for elem in vals_to_print]

  print(','.join(to_print), file=stream)


def train_vae(dataset, config={}, vae_affix="", output_filepath=""):
  '''
  Method to train a VAE with a dataset.

  :param dataset: A Dataset object or a Numpy array containing a list of 3-channel 64x64 images
  :param config: A configuration dictionary. Available keys are "z_size", "batch_size", "learning_rate", "kl_tolerance", \
    "num_epoch"
  :param vae_affix: The string to append to the end of the VAE weights filename.
  :param output_file: The filename of output file.
  :returns vae: The VAE object, can be used to visualize end-product.
  '''

  # Hyperparameters for ConvVAE

  z_size = config["z_size"] if "z_size" in config else 32
  batch_size = config["batch_size"] if "batch_size" in config else 100
  learning_rate = config["learning_rate"] if "learning_rate" in config else 0.0001
  kl_tolerance = config["kl_tolerance"] if "kl_tolerance" in config else 0.5
  num_epoch = config["num_epoch"] if "num_epoch" in config else 100

  model_save_path = "tf_vae"
  if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
  
  if len(output_filepath) > 0:
    output_filestream = open(output_filepath, "w")
  else:
    output_filestream = sys.stdout

  filename = "vae.json" if not vae_affix else f"vae_{vae_affix}.json"
  filepath = os.path.join("tf_vae", filename)

# split into batches:
  total_length = len(dataset)
  num_batches = total_length//batch_size

  reset_graph()

  vae = ConvVAE(z_size=z_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                kl_tolerance=kl_tolerance,
                is_training=True,
                reuse=False,
                gpu_mode=True)

  _print_header(num_epoch, z_size, output_filestream)

  for epoch in range(num_epoch):
    for idx in range(num_batches):
      batch = dataset[idx*batch_size:(idx+1)*batch_size]

      obs = batch.astype(np.float)/255.0

      feed = {vae.x: obs,}

      stats = vae.sess.run([
        vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op, vae.other_losses
      ], feed)

      train_step = stats[3]
    
      if ((train_step+1) % 1000 == 0):
        _print_stats(stats, output_filestream)

  # finished, final model:
  vae.save_json(filepath)
  if output_filestream != sys.stdout:
    output_filestream.close()
  
  return vae


if __name__ == "__main__":
  for z in [4]:
    config = {"z_size": z, "num_epoch": 10}
    dataset = create_dataset(DATA_DIR)
    lbl = str(config["z_size"])
    affix = f"z_{lbl}"
    vae = train_vae(dataset, config, affix, f"./{z}.txt")
    visualize(vae, dataset, num_random_imgs=10)
