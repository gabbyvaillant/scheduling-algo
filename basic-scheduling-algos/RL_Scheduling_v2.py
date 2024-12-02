#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pynvml
import subprocess
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
class MatrixMultiplicationJob:
    def __init__(self, matrix_size, job_name):
        self.matrix_size = matrix_size
        self.job_name = job_name

    def get_command(self):
        return f"python run_matrix_multiplication.py --matrix_size {self.matrix_size}"


class DeepLearningJob:
    def __init__(self, epochs, job_name):
        self.epochs = epochs
        self.job_name = job_name

    def get_command(self):
        return f"python train_deep_learning.py --epochs {self.epochs}"


class XGBoostJob:
    def __init__(self, n_estimators, max_depth, job_name):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.job_name = job_name

    def get_command(self):
        return f"python train_xgboost.py --n_estimators {self.n_estimators} --max_depth {self.max_depth}"


class LSTMJob:
    def __init__(self, sequence_length, epochs, job_name):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.job_name = job_name

    def get_command(self):
        return f"python train_lstm.py --sequence_length {self.sequence_length} --epochs {self.epochs}"

job_queue = deque()
# Add jobs to the queue
job1 = MatrixMultiplicationJob(matrix_size=30000, job_name="MatrixMultiplication_Job1")
job2 = DeepLearningJob(epochs=5, job_name="DL_Job1")
job3 = XGBoostJob(n_estimators=1200, max_depth=30, job_name="XGBoost_Job1")
job4 = LSTMJob(sequence_length=200, epochs=10, job_name="LSTM_Job1")
        
# Helper functions for job scheduler
def get_pending_jobs():
    return list(job_queue)  # Return all jobs in the queue

def submit_job(gpu_id, job_id):
    try:
        command = job_id.get_command()  # Get the job's command
        print(f"Submitting job {job_id.job_name} to GPU {gpu_id} with command: {command}")
        subprocess.run(command, shell=True, check=True)

        # Remove the job from the queue after submission
        job_queue.remove(job_id)

    except Exception as e:
        print(f"Error submitting job {job_id.job_name} to GPU {gpu_id}: {e}")

# Custom Gym Environment
class RealGpuSchedulerEnv(gym.Env):
    def __init__(self, max_steps=100):
        super(RealGpuSchedulerEnv, self).__init__()
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.max_steps = max_steps
        self.current_step = 0
        # Store job executions times
        self.job_history = {gpu_id: [] for gpu_id in range(self.device_count)}
        # State: GPU utilizations and number of pending jobs
        self.observation_space = spaces.Box(
            low=np.array([0.0] * self.device_count + [0.0], dtype=np.float32),
            high=np.array([1.0] * self.device_count + [np.inf], dtype=np.float32),
            shape=(self.device_count + 1,),
            dtype=np.float32
        )
        # Actions: Assign to GPU 0, GPU 1, ..., GPU N-1, or delay
        self.action_space = spaces.Discrete(self.device_count + 1)
        self._seed = None
    
    def step(self, action):
        # Get current GPU utilizations
        utilizations = [pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu / 100.0 for i in range(self.device_count)]
        
        # Get pending jobs
        jobs = get_pending_jobs()
        num_pending = len(jobs)
        
        reward = 0
        done = False
        # Assume no truncation initially
        truncated = False
        jobs_completed = 0
        
        if action < self.device_count:
            if jobs:
                job_id = jobs.pop(0)
                start_time = time.time()
                submit_job(action, job_id)
                end_time = time.time()
                self.job_history[action].append((job_id.job_name, start_time, end_time))
                jobs_completed += 1
                # After submission, fetch new utilization
                power_draws = [pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(self.device_count)]
                total_power = sum(power_draws)
                performance = jobs_completed
                reward = performance  # Higher reward for better efficiency  # Penalize high utilization
            else:
                reward = -0.01  # Small penalty if no jobs to assign
        else:
            # Action to delay, small penalty
            reward = -0.05
        
        # Update state
        utilizations = [pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu / 100.0 for i in range(self.device_count)]
        num_pending = len(get_pending_jobs())
        state = np.array(utilizations + [num_pending], dtype=np.float32)
        
        # Define a condition for episode termination
        if num_pending == 0:
            done = True
        # Check if the episode is truncated (e.g. max steps reached)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        
        return state, reward, done, truncated, {}

    def render_gantt_chart(self, filename="gantt_chart.png"):
        # Create a Gantt chart to show job schedules
        fig, ax = plt.subplots(figsize=(10, 6))
        for gpu_id, jobs in self.job_history.items():
            for job_name, start_time, end_time in jobs:
                ax.barh(f"GPU {gpu_id}", end_time - start_time, left=start_time, label=job_name)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GPU")
        ax.set_title("Gantt Chart of GPU Job Scheduling")
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left")
        plt.tight_layout()
        plt.savefig(filename) # Save the chart to a file
        print(f"Gantt chart saved as {filename}")
    
    def reset(self, seed=None, options=None):
        # Set the random seed for reproducible results
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        # Initialize GPU utilizations (normalized to [0,1])
        utilizations = [pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu / 100.0 for i in range(self.device_count)]
        # Number of pending jobs in the queue
        num_pending = len(get_pending_jobs())
        # Combine utilizations and pending jobs into the initial observation
        initial_observation = np.array(utilizations + [num_pending], dtype=np.float32)
        # Return observation and additional information (a dict here as required by gym)
        return initial_observation, {}
    
    def render(self, mode='human'):
        utilizations = [pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu for i in range(self.device_count)]
        power_draws = [pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(i)) / 1000.0 for i in range(self.device_count)]
        num_pending = len(get_pending_jobs())
        print(f"GPU Utilizations: {utilizations}, GPU Power Usage: {power_draws}, Pending Jobs: {num_pending}")

    def close(self):
        pynvml.nvmlShutdown()


# Add jobs to the queue
job_queue.extend([job1, job2, job3, job4])

# Initialize environment and check it
env = RealGpuSchedulerEnv()
check_env(env)

# Initialize the RL agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
print("Starting training...")
model.learn(total_timesteps=10000)
print("Training completed.")

# Save the model
model.save("real_gpu_scheduler_ppo")
print("Model saved.")

# Load the model (optional)
model = PPO.load("real_gpu_scheduler_ppo")

# Deploy the agent
print("Deploying the agent...")
obs, _ = env.reset() # Extract only the observation
done = False
while not done:
    # Use the extracted obs for model.predict()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    env.render()
    env.render_gantt_chart()

env.close()
