'''
This module allows experiment on different z_size (to see which is better) and redirects the outputs into separate text files. 
The module is not very significant, can ignore this.
'''

from vae_train import train_vae
from data_loader import create_dataset
from visualize import visualize

DATA_DIR = 'data'

def experiment_z_size():
    list_of_z_size = [4, 8, 16, 32]
    config = {}
    dataset = create_dataset(DATA_DIR)

    for z_size in list_of_z_size:
        config["z_size"] = z_size
        config["num_epoch"] = 30
        affix = f"z_{z_size}"
        vae = train_vae(dataset, config, affix, f"./z_{z_size}.txt")
        visualize(vae, dataset)

if __name__ == "__main__":
    experiment_z_size()