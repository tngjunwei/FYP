'''
This module allows visualization of the decoded images, in comparison to the original.
'''

import numpy as np
import cv2
import os

def visualize(vae, dataset, num_random_imgs=5, fixed=True):
    '''
    Method to visualize the images.
    
    :param vae: A VAE object
    :param dataset: A Dataset object or a Numpy array containing a list of 3-channel 64x64 images
    :param num_random_imgs: Number of randomly sampled images to visualize
    :param fixed: Boolean to indicate whether the visualized frames are always the same or randomized
    '''

    def encode_batch(batch_img):
        simple_obs = np.copy(batch_img).astype(np.float)/255.0
        simple_obs = simple_obs.reshape(-1, 64, 64, 3)
        mu, logvar = vae.encode_mu_logvar(simple_obs)
        z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))
        return mu, logvar, z

    def decode_batch(batch_z):
        # decode the latent vector
        batch_img = vae.decode(batch_z.reshape(-1, vae.z_size)) * 255.
        batch_img = np.round(batch_img).astype(np.uint8)
        batch_img = batch_img.reshape(-1, 64, 64, 3)
        return batch_img


    if fixed:
        random_idx_for_batch = np.arange(vae.batch_size)
    else:
        random_idx_for_batch = np.random.choice(np.arange(0, len(dataset), dtype=np.int64), size=vae.batch_size, replace=False)
    dataset_sub = dataset[random_idx_for_batch] #choose random files
    mu, logvar, z = encode_batch(dataset_sub)
    decoded = decode_batch(z)

    if fixed:
        chosen_idx = np.arange(0, len(dataset_sub))
    else:
        chosen_idx = np.random.choice(np.arange(0, len(dataset_sub), dtype=np.int64), size=num_random_imgs, replace=False)
    
    dataset_sub = dataset_sub[chosen_idx]
    decoded = decoded[chosen_idx]

    dataset_sub = dataset_sub.reshape((-1, 64, 3))
    decoded = decoded.reshape((-1, 64, 3))
    img = np.concatenate([dataset_sub, decoded], axis=1)
    cv2.imwrite(os.path.join('img', f'img_{vae.z_size}.png'), img)

def visualize_all(vae, dataset):
    '''
    Method to visualize the images.
    
    :param vae: A VAE object
    :param dataset: A Dataset object or a Numpy array containing a list of 3-channel 64x64 images
    '''

    def encode_batch(batch_img):
        simple_obs = np.copy(batch_img).astype(np.float)/255.0
        simple_obs = simple_obs.reshape(-1, 64, 64, 3)
        mu, logvar = vae.encode_mu_logvar(simple_obs)
        z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))
        return mu, logvar, z

    def decode_batch(batch_z):
        # decode the latent vector
        batch_img = vae.decode(batch_z.reshape(-1, vae.z_size)) * 255.
        batch_img = np.round(batch_img).astype(np.uint8)
        batch_img = batch_img.reshape(-1, 64, 64, 3)
        return batch_img

    original = []
    decoded_imgs = []
    batch_size = vae.batch_size
    num_batches = len(dataset) // batch_size
    print(num_batches, len(dataset))
    for i in range(num_batches):
        batch = dataset[i*batch_size:(i+1)*batch_size]
        mu, logvar, z = encode_batch(batch)
        decoded = decode_batch(z)

        original.append(batch)
        decoded_imgs.append(decoded)
    
    original_series = np.array(original).reshape((10, -1, 64, 64, 3))
    print(original_series.shape)
    decoded_series = np.array(decoded_imgs).reshape((10, -1, 64, 64, 3))

    for i in range(2):
        folder_path = os.path.join("img", str(i))
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        for j in range(1500):
            original_frame = original_series[i, j]
            decoded_frame = decoded_series[i, j]
            frame = np.concatenate([original_frame, decoded_frame], axis=1)
            save_path = os.path.join(folder_path, f"{j}.png")
            cv2.imwrite(save_path, frame)



if __name__ == "__main__":
    from vae import ConvVAE
    from data_loader import create_dataset

    DATA_DIR = "data"

    #list_of_z = [4, 8, 16, 32]
    list_of_z = [4]
    list_of_vae = []
    
    # Load VAEs
    for z in list_of_z:
        filename = f"vae_z_{z}.json"
        filepath = os.path.join("tf_vae", filename)
        vae = ConvVAE(z_size=z, batch_size=100, gpu_mode=True)
        vae.load_json(filepath)
        list_of_vae.append(vae)

    dataset = create_dataset(DATA_DIR)
    for vae in list_of_vae:
        #visualize(vae, dataset, num_random_imgs=8)
        visualize_all(vae, dataset)
