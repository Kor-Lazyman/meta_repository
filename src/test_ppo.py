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
from ppo_only import *
from config_folders import *
from log_RL import *
from log_SimPy import *
# Now under developing Code

start_time = time.time()
main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
graph_writer = SummaryWriter(log_dir=GRAPH_LOG)
# config rl에서 DRL_TENSORBOARD를 true로 할 것
total_episodes = N_EPISODES  
episode_counter = 0

test_time_lst = []
env_main = GymInterface()
model = PPOAgent(len(env_main.reset()), [len(ACTION_SPACE) for _ in range(MAT_COUNT)], N_EPISODES)
model = inner_model(env_main, model.policy.state_dict(),10, 3e-4, 20, N_EPISODES)
model.main_module()
state = env_main.reset()
episode_reward = 0
print("\n\nTEST SCENARIO: ", env_main.scenario,"\n")
graph_dict = {}
for x in range(SIM_TIME):
    action = model.select_action(state)
    state, reward, done, _ = env_main.step(action[0])
    episode_reward += reward
    
    for inven in env_main.inventoryList:
        graph_writer.add_scalar(f"Onhand/{I[inven.item_id]['NAME']}", STATE_DICT[-1][f"On_Hand_{I[inven.item_id]['NAME']}"], x)
        graph_dict[f"Onhand/{I[inven.item_id]['NAME']}"] = STATE_DICT[-1][f"On_Hand_{I[inven.item_id]['NAME']}"]
    for id in range(len(action[0])):
        graph_writer.add_scalar(f"Order_Material/Material{id}",action[0][id], x)
        graph_dict[f"Order_Material/Material{id}"] = action[0][id]
    
    graph_writer.add_scalar(f"Demand",I[0]["DEMAND_QUANTITY"], x)
    graph_dict[f"Demand"] = I[0]["DEMAND_QUANTITY"]
    graph_writer.add_scalars("Final_Results", graph_dict,x)

print(f"Time:{time.time()-start_time}")
print(test_time_lst)