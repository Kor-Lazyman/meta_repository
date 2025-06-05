# === Import necessary modules and configuration files ===
import GymWrapper as gw
import time
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
from Def_Scenarios import *
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
import multiprocessing
from meta_model import *
from torch.utils.tensorboard import SummaryWriter
from config_folders import *

# === Hyperparameters ===
K = N_EPISODES
ALPHA = 0.0001  # Inner loop step size (unused)
BATCH_SIZE = 20
N_STEPS = SIM_TIME * K
BETA = 0.0001  # Outer loop learning rate
train_scenario_batch_size = 5
test_scenario_batch_size = 2
num_outer_updates = 1000  # Total meta-training iterations

# === Meta-Learner class for meta-training using outer loop ===
class MetaLearner:
    def __init__(self, env, policy='MlpPolicy', alpha=ALPHA, beta=BETA):
        """
        Initialize meta-learning framework.
        """
        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.beta = beta
        self.action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)]
        self.state_dim = len(self.env.reset())

        # Logging
        self.graph_writer = SummaryWriter(log_dir=GRAPH_LOG)
        self.writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)

        # Meta-policy model (outer loop)
        self.meta_model = outer_loop(
            state_dim=self.state_dim,
            action_dims=self.action_dims,
            total_time_steps=num_outer_updates,
            lr=1e-5,
            gamma=0.99,
            clip_epsilon=CLIP_EPSILON,
            update_steps=5
        )

    def meta_update(self, adapted_models):
        """
        Perform meta-update using adapted models (inner-loop models).
        """
        print("MetaLearning Start")
        self.meta_model.update_steps = 10
        self.meta_model.device = DEVICE
        for model in adapted_models:
            self.meta_model.update(model)

    def meta_test(self, iteration, test_scenario_batch):
        """
        Evaluate the current meta-policy on test scenarios.
        """
        print("Meta Learning Started")
        count_case = 0
        all_rewards = []

        for test_scenario in test_scenario_batch:
            count_case += 1
            state = self.env.reset()
            self.env.scenario = test_scenario
            episode_reward = 0
            print("\n\nTEST SCENARIO: ", self.env.scenario, "\n")
            graph_dict = {}

            for x in range(SIM_TIME):
                action = self.meta_model.select_action(state)
                state, reward, done, _ = self.env.step(action[0])
                episode_reward += reward

                # Log to TensorBoard only for the last test scenario at final iteration
                if count_case == len(test_scenario_batch) and GRAPH_WRITER and iteration == num_outer_updates:
                    for inven in self.env.inventoryList:
                        self.graph_writer.add_scalar(f"Onhand/{I[inven.item_id]['NAME']}", STATE_DICT[-1][f"On_Hand_{I[inven.item_id]['NAME']}"], x)
                        graph_dict[f"Onhand/{I[inven.item_id]['NAME']}"] = STATE_DICT[-1][f"On_Hand_{I[inven.item_id]['NAME']}"]
                    for id in range(len(action[0])):
                        self.graph_writer.add_scalar(f"Order_Material/Material{id}", action[0][id], x)
                        graph_dict[f"Order_Material/Material{id}"] = action[0][id]
                    self.graph_writer.add_scalar(f"Demand", I[0]["DEMAND_QUANTITY"], x)
                    graph_dict[f"Demand"] = I[0]["DEMAND_QUANTITY"]
                    self.graph_writer.add_scalars("Final_Results", graph_dict, x)

            all_rewards.append(episode_reward)

        # Compute mean and std dev reward
        meta_mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        self.log_to_tensorboard(iteration, meta_mean_reward, std_reward)

        return meta_mean_reward, std_reward

    def log_to_tensorboard(self, iteration, mean_reward, std_reward):
        """
        Log reward statistics to TensorBoard.
        """
        self.writer.add_scalar("Reward/Mean", mean_reward, iteration)
        self.writer.add_scalar("Reward/Std", std_reward, iteration)


# === Inner model used for task-specific adaptation ===
class inner_model:
    def __init__(self, env, state_dict, n_steps, lr, batch_size, total_steps):
        self.env = env
        self.action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)]
        self.state_dim = len(self.env.reset())
        self.step = n_steps
        self.lr = lr
        self.batch_szie = batch_size
        self.total_steps = total_steps
        self.episode_counter = 0
        self.build_model(state_dict)

    def build_model(self, state_dict):
        """
        Initialize PPO agent using policy state_dict.
        """
        self.model = PPOAgent(
            state_dim=self.state_dim,
            action_dims=self.action_dims,
            total_time_steps=self.total_steps,
            lr=self.lr,
            gamma=0.99,
            clip_epsilon=CLIP_EPSILON,
            update_steps=self.step
        )
        self.model.policy.load_state_dict(state_dict)

    def simulation_worker(self, core_index, model_state_dict):
        """
        Simulate an episode (used in multiprocessing).
        """
        state = env.reset()
        agent = self.build_model(model_state_dict)
        agent.policy.load_state_dict(model_state_dict)

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
        Process transitions into separate arrays for PPO update.
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
        Helper for multiprocessing map.
        """
        return self.simulation_worker(*args)

    def update_model(self, states, actions, rewards, next_states, dones, log_probs):
        """
        Store transitions and perform PPO update.
        """
        for j in range(len(states)):
            self.model.store_transition((states[j], actions[j], rewards[j], next_states[j], dones[j], log_probs[j]))
        
        _, self.memory = self.model.update()
        self.episode_counter += 1
        print(f"Completed {self.episode_counter} / {self.total_steps} episodes.")

# === Multiprocessing wrapper to run one episode ===
def worker_wrapper(model_state_dict):
    """
    Run a simulation for one worker.
    """
    env = GymInterface()
    state = env.reset()
    agent = inner_model(env, model_state_dict, n_steps=10, lr=1e-5, batch_size=BATCH_SIZE, total_steps=200)

    done = False
    episode_transitions = []
    episode_reward = 0

    while not done:
        action, log_prob = agent.model.select_action(state)
        next_state, reward, done, info = env.step(action)
        episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
        episode_reward += reward
        state = next_state

    return (episode_transitions, episode_reward)


# === Main execution: Meta-training process ===
if __name__ == '__main__':
    start_time = time.time()
    env = GymInterface()
    meta_learner = MetaLearner(env)

    # Generate scenarios for meta-learning
    all_scenarios = create_scenarios()
    train_scenarios, test_scenarios = split_scenarios(all_scenarios)
    print(f"Total {len(all_scenarios)} scenarios generated.")
    print(f"Train: {len(train_scenarios)}, Test: {len(test_scenarios)}")

    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)

    # === Meta-training loop ===
    for iteration in range(num_outer_updates):
        env.scenario_batch_size = train_scenario_batch_size
        train_scenario_batch = random.sample(train_scenarios, train_scenario_batch_size)

        if iteration == num_outer_updates - 1:
            meta_learner.env.outer_end = True

        adapted_models = []
        for scenario in train_scenario_batch:
            # === Inner Loop Adaptation ===
            meta_learner.env.scenario = scenario
            adapted_model = inner_model(meta_learner.env, meta_learner.meta_model.policy.state_dict(),
                                        n_steps=10, lr=1e-5, batch_size=BATCH_SIZE, total_steps=100)

            episode_counter = 0
            start_time = time.time()

            # Parallel episode simulation and update
            while episode_counter < adapted_model.total_steps:
                batch_workers = min(N_MULTIPROCESS, adapted_model.total_steps - episode_counter)
                model_state_dict = adapted_model.model.policy.state_dict()
                tasks = [model_state_dict for _ in range(batch_workers)]

                for result in pool.imap_unordered(worker_wrapper, tasks):
                    transitions, _ = result
                    states, actions, rewards, next_states, dones, log_probs = adapted_model.process_transitions([transitions])
                    adapted_model.update_model(states, actions, rewards, next_states, dones, log_probs)
                    episode_counter += 1

            print("Inner_Loop_Ended: ", time.time() - start_time)
            adapted_models.append(adapted_model)

            env.cur_episode = 1
            env.cur_inner_loop += 1

        # === Outer Loop Meta-Update ===
        meta_learner.meta_update(adapted_models)

        # === Meta-Test ===
        test_scenario_batch = random.sample(train_scenario_batch, test_scenario_batch_size)
        meta_mean_reward, meta_std_reward = meta_learner.meta_test(iteration, test_scenario_batch)
        print(f'Iteration {iteration+1}/{num_outer_updates} - Mean Reward: {meta_mean_reward:.2f} ± {meta_std_reward:.2f}\n')
        print('===========================================================')

        # Reset environment counters
        env.cur_episode = 1
        env.cur_inner_loop = 1
        env.cur_outer_loop += 1
        env.outer_end = False

        # Save meta-policy model
        if EXPERIMENT_MAML == False:
            torch.save(meta_learner.meta_model.policy.state_dict(), SAVED_MODEL_NAME)
        else:
            torch.save(meta_learner.meta_model.policy.state_dict(), os.path.join(TENSORFLOW_LOGS, SAVED_MODEL_NAME))

    training_end_time = time.time()
    print("\nMETA TRAINING COMPLETE\n")

    # === Final evaluation ===
    meta_mean_reward, meta_std_reward = meta_learner.meta_test(num_outer_updates, test_scenario_batch)
    print(f"Final Mean Reward: {meta_mean_reward:.2f} ± {meta_std_reward:.2f}")

    # === Timing info ===
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes")
    print(f"Training time: {(training_end_time - start_time)/60:.2f} minutes")
    print(f"Test time: {(end_time - training_end_time)/60:.2f} minutes")

    env.render()
    pool.close()
    pool.join()
