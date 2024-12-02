# Optimizing GPU Scheduling for Machine Learning Workloads ⏱️🖥️
This project explores scheduling algorithms for deep learning tasks, focusing on optimizing GPU utilization. We compare basic scheduling methods across four deep learning tasks under heavy GPU load, then introduce a reinforcement learning-based approach to improve GPU efficiency. Unlike traditional methods that focus on CPU usage or makespan, our approach prioritizes GPU usage, which is crucial as GPUs are increasingly used to accelerate deep learning. To address the lack of datasets with GPU usage information, we created a set of synthetic deep learning and machine learning tasks to test and refine the algorithms.


## 1. Set up experiment on Cloud Lab
This project uses Cloudlab for GPU access. Students in AMS 560 will have access through this link: https://www.cloudlab.us/show-project.php?project=AMS560-SBU

Check to see the resource availability: https://www.cloudlab.us/resinfo.php

Check to see which hardwares have access to GPUs: https://docs.cloudlab.us/hardware.html

Start an experiment with the following choices:
- Profile: Open-Stack
- Hardware type: Cloudlab Wiscosin -> c240g5 (or another hardware with at least 1 GPU)
- Number of compute nodes (at Site 1) -> 0
- Schedule experiment for a time where there is resource availability 

Open terminal and ssh into node 

```bash
#EXAMPLE:
ssh vailg@c240g5-110219.wisc.cloudlab.us


```
## 2. Clone this repository 


```bash
# Install git on node

sudo apt update
sudo apt install git

#Check to see if it properly installed
git --version

#Clone this repository
git clone https://github.com/gabbyvaillant/scheduling-algo.git

```

## 3. Set up GPU driver

```bash
sudo apt install ubuntu-drivers-common

sudo ubuntu-drivers list
# check if ubuntu-drivers install successfully

sudo ubuntu-drivers install

sudo reboot
# reboot the system after ubuntu-drivers install
# you will loss connection and this step need some time

# Wait a few minutes for it to reboot and then reconnect with your system and check if NVIDIA drivers install successfully or not
nvidia-smi

```
You should get something like the following output: 
![Alt Text]([scheduling-algo/screenshots
/gpu-setup.png](https://github.com/gabbyvaillant/scheduling-algo/blob/main/screenshots/gpu-setup.png))


## 4. Create a virtual enviornment to run the machine learning tasks

Install miniconda to create a virtual enviornment on the node 
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda init --all

```
You must exit the node by the command 
```bash exit ```
and re ssh into the node for the changes to work (or maybe rebooting would be better)

```bash
conda create --name tf_gpu python=3.8

conda activate tf_gpu

conda install pip

python3 -m pip install 'tensorflow[and-cuda]'

#pip install tensorflow==2.7

conda install -c conda-forge cudatoolkit=11.5 cudnn=8.3

#Check version of cuda complier
nvcc --version

```

Run the check_gpu.py script to see if tensorflow and cude recognizes the GPU

```bash

python3 check_gpu.py

```
To save and close use hit Shift+control X and hit enter to save

Run the file

```bash

python3 check_gpu.py

```
The output should say that there are 1 GPU available. 


## 5. Run baseline scheduling algorithms (FCFS, Round-Robin, SJF) on ML tasks

Go to /cloudlab-code/baseline/ to find the files for the Python code for each baseline algorithm.
 

## 6. Run reinforcement learning based scheduling algorithms on tasks

Go to /cloudlab-code/RF/ to find the file for the reinforcement learning scheduling algorithm.




---

## RUNNING GPU SIMULATIONS 

For the Reinforcement learning model, running the model prediction code will simulate GPU scheduling using the trained RL model.
Load the model and run the code in "...".


Ex: Evaluation function and task generator to simulate tasks:

Task Generator:
```python
# Task generation function
def generate_task(task_id):
    task_types = {
        "image_classification": {
            "gpu_demand": (20, 40), "memory_demand": (10, 30), "duration": (5, 10), "priority": (1, 3)
        },
        "lstm_training": {
            "gpu_demand": (30, 50), "memory_demand": (40, 60), "duration": (7, 15), "priority": (2, 4)
        },
        "cnn_training": {
            "gpu_demand": (50, 80), "memory_demand": (30, 50), "duration": (10, 20), "priority": (3, 5)
        }
    }
    task_type = random.choice(list(task_types.keys()))
    params = task_types[task_type]
    gpu_demand = random.randint(*params["gpu_demand"])
    memory_demand = random.randint(*params["memory_demand"])
    duration = random.randint(*params["duration"])
    priority = random.randint(*params["priority"])

    return {
        "id": task_id,
        "type": task_type,
        "gpu_demand": gpu_demand,
        "memory_demand": memory_demand,
        "duration": duration,
        "priority": priority,
        "remaining_duration": duration,
        "start_time": None
    }
```


### Evaluation function:
```python
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def evaluate_model(model, env, num_tasks=200):
    # Initialize the environment with a new set of tasks
    env.tasks = [generate_task(i) for i in range(num_tasks)]

    # Reset the environment to start fresh
    obs = env.reset()

    # Track rewards and metrics
    rewards = []
    total_waiting_time = 0
    total_time = 0

    while True:
        # Model predicts the best GPU to schedule the task
        action, _ = model.predict(obs, deterministic=True)

        # Perform one step in the environment
        obs, reward, done, _ = env.step(action)

        # Track the reward for this step
        rewards.append(reward)

        # Track the total waiting time (sum of all task waiting times)
        total_waiting_time += sum(env.task_waiting_times)

        # Render the environment to visualize the current scheduling (Gantt chart)
        env.render()

        # Break the loop if all tasks are completed
        if done:
            break

        # Update total time (tracking the makespan)
        total_time += 1

    # Compute average waiting time
    avg_waiting_time = total_waiting_time / num_tasks

    print(f"Final Makespan (Total Time): {env.total_time}")
    print(f"Average Waiting Time: {avg_waiting_time:.2f}")
    print(f"Total Reward: {sum(rewards)}")

    return env.total_time, avg_waiting_time
```

### Run evaluation on model:

```python


env = MLWorkloadSchedulingEnv(num_gpus=2)  # Initialize the environment with 2 GPUs


makespan, avg_waiting_time = evaluate_model(model, env, num_tasks=50)

# 4. Print the evaluation results
print(f"Final Makespan (Total Time): {makespan}")
print(f"Average Waiting Time: {avg_waiting_time:.2f}")

```

This project was created for AMS 560.
Contributors: Gabrielle Vaillant, Michael Deisler, Iftekhar Alam
