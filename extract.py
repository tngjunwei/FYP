'''
This module is for extracing environment frames mainly for Pong (also for Enduro, Pacman, and Breakout)
'''
import gym
import cv2
import numpy as np
import random

# Preprocessing info (crop dim)
#obs = prepro(obs, start_height=35, end_height=190, start_width=5, end_width=None) #pong
#obs = prepro(obs, start_height=20, end_height=300, start_width=0, end_width=200) #original breakout
#obs = prepro(obs, start_height=33, end_height=198, start_width=8, end_width=152) #modified breakout (better)
#obs = prepro(obs, start_height=60, end_height=155, start_width=8, end_width=None) #enduro
#obs = prepro(obs, start_height=1, end_height=171) #mspacman


def prepro(I, start_height=None, end_height=None, start_width=None, end_width=None):
    I = I[start_height:end_height, start_width:end_width] # crop
    I = cv2.resize(I, (64, 64))
    return I

NUM_TRIALS = 1000
NUM_EPS = 1000


def get_pong_frames():
    env = gym.make("Pong-v0")
    env.render('rgb_array')

    for _ in range(NUM_TRIALS):
        obs = env.reset()
        actions = []
        frames = []
        for _ in range(NUM_EPS):
            action = random.randint(0, 4) # For other envs, you might want to do "env.action_space.sample()" instead
            obs, reward, done, info = env.step(action)

            obs = prepro(obs, start_height=35, end_height=190, start_width=5, end_width=None) # change this line according to env

            frames.append(obs)
            actions.append(action)

            if done:
                np.savez_compressed(f'./record/{random.randint(1, 99999999)}.npz', obs=frames, action=actions)
                break
        np.savez_compressed(f'./record/{random.randint(1, 99999999)}.npz', obs=frames, action=actions)

    env.close()
