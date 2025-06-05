import GymWrapper as gw
import os
import time
import torch.multiprocessing as mp
import math
import torch
from GymWrapper import GymInterface 
from PPO import PPOAgent
from config_RL import *
from torch.utils.tensorboard import SummaryWriter

class inner_model:
    def __init__(self, env, state_dict, n_steps, lr, batch_size, total_steps):
        self.action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)]  # MultiDiscrete
        self.env = env
        self.step = n_steps
        self.lr = lr
        self.batch_szie = batch_size
        self.state_dim = len(self.env.reset())
        self.total_steps = total_steps
        self.build_model(state_dict)

    def build_model(self, state_dict):
        """
        Build and return a PPOAgent model using the environment's state dimension and MAT_COUNT.
        
        Args:
            env (GymInterface): The Gym environment instance.
            
        Returns:
            PPOAgent: A PPO agent instance initialized with proper hyperparameters.
        """
        self.model = PPOAgent(
            state_dim = self.state_dim,
            action_dims = self.action_dims,
            total_time_steps = self.total_steps,
            lr = self.lr,
            gamma = 0.99,
            clip_epsilon = CLIP_EPSILON,
            update_steps = self.step
        )
        self.model.policy.load_state_dict(state_dict)

    def simulation_worker(self, core_index, model_state_dict):
        """
        Run a single episode in a worker process and return the transitions and total reward.
        
        Args:
            core_index: The index of the worker process.
            model_state_dict: The state dictionary of the main model.
            
        Returns:
            tuple: (core_index, episode_transitions, episode_reward)
        """
        env = GymInterface()
        agent = self.build_model(model_state_dict)
        agent.policy.load_state_dict(model_state_dict)
        
        state = env.reset()
        done = False
        episode_transitions = []
        episode_reward = 0
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
            episode_reward += reward
            state = next_state
            return (core_index, episode_transitions, episode_reward)

    def process_transitions(self, transitions):
        """
        Combine and unpack transition data collected from a worker.
        
        Args:
            transitions (list): A list of transition lists from one or more workers.
        
        Returns:
            tuple: Separate lists for states, actions, rewards, next_states, dones, and log_probs.
        """
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
        for worker_transitions in transitions:
            for tr in worker_transitions:
                states.append(tr[0])
                actions.append(tr[1])
                rewards.append(tr[2])
                next_states.append(tr[3])
                dones.append(tr[4])
                log_probs.append(tr[5])

        return states, actions, rewards, next_states, dones, log_probs

    def worker_wrapper(self, args):
        """
        Wrapper function for simulation_worker to unpack arguments.
        
        Args:
            args (tuple): A tuple containing (core_index, model_state_dict)
            
        Returns:
            tuple: The result of simulation_worker.
        """
        return self.simulation_worker(*args)
    
    def main_module(self):
        
        if __name__ == '__main__':
            mp.set_start_method('spawn')
            pool = mp.Pool(processes = N_MULTIPROCESS)
            print("Check")
            time.sleep(1000)
            episode_counter = 0
            #self.model.policy.load_state_dict(torch.load(state_dict))
            start_time = time.time()

            while episode_counter < self.total_steps:
                batch_workers = min(N_MULTIPROCESS, self.total_steps - episode_counter)
                model_state_dict = self.model.policy.state_dict()
                
                tasks = [(i, model_state_dict) for i in range(batch_workers)]

                for result in pool.imap_unordered(self.worker_wrapper, tasks):  # imap_unordered: 워커들의 결과를 먼저 끝난 순서대로 반환환
                    _, transitions, _ = result

                    states, actions, rewards, next_states, dones, log_probs = self.process_transitions([transitions])
                    for j in range(len(states)):
                        self.model.store_transition((states[j], actions[j], rewards[j], next_states[j], dones[j], log_probs[j]))
                    
                    _,  self.memory = self.model.update()
                    episode_counter += 1
                    print(f"Completed {episode_counter} / {self.total_steps} episodes.")
                    print(self.memory)
                print("Check",self.last_memory)
            
            if SAVE_MODEL:
                model_path = os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME)
                torch.save(self.model.policy.state_dict(), model_path)
                print(f"{SAVED_MODEL_NAME} saved successfully")
            
            end_time = time.time()
            computation_time = (end_time - start_time) / 60
            print(f"Total computation time: {computation_time:.2f} minutes")

            return self.model.policy.state_dict()