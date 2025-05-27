import GymWrapper as gw
import os
import time
import multiprocessing
import math
import torch
from GymWrapper import GymInterface 
from PPO import PPOAgent
from config_RL import *
from torch.utils.tensorboard import SummaryWriter
from inner_Model import *
main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)

# config rl에서 DRL_TENSORBOARD를 true로 할 것
total_episodes = N_EPISODES  
episode_counter = 0
env_main = GymInterface()
model = PPOAgent(len(env_main.reset()), [len(ACTION_SPACE) for _ in range(MAT_COUNT)])
model = inner_model(env_main, model.policy.state_dict(),10, 3e-4, 20, N_EPISODES)
model.main_module()