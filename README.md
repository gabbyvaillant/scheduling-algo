# Optimizing GPU Scheduling for Machine Learning Workloads ⏱️🖥️
This project compares scheduling algorithms for machine learning tasks, focusing on optimizing GPU usage and power utilization. We evaluate fundamental static scheduling algorithms across various tasks under heavy GPU load and subsequently test a reinforcement learning-based scheduling algorithm to enhance GPU performance. Unlike prior studies, this project emphasizes GPU usage and power efficiency, addressing the practical challenges faced by large data centers and the increasing dependence on GPUs for accelerating deep learning processes.

To contribute to the growing field of reinforcement learning-based scheduling algorithms, we implemented one approach. This work aligns with recent advancements in reinforcement learning applied to resource optimization.

To overcome the scarcity of datasets with GPU usage information, we created a custom set of deep learning tasks and computationally intensive operations—to assess the algorithms' performance in terms of GPU usage and efficiency.


We have two options for this project. 
1. Using cloud lab to work with one single GPU or multiple GPUs (harder to get access)
2. Simulating multiple GPUs in an enviornment (due to the lack of resources)


# Cloudlab for GPU access
## 1. Set up experiment on Cloud Lab
This project uses Cloudlab for GPU access. Students in AMS 560 will have access through this link: https://www.cloudlab.us/show-project.php?project=AMS560-SBU

Check to see the resource availability: https://www.cloudlab.us/resinfo.php

Check to see which servers have access to GPUs: https://docs.cloudlab.us/hardware.html

DEPENDING ON YOUR GOAL, CHOOSE WHICH SCENARIO YOU WOULD LIKE TO TEST: 

Single GPU (1.a.) vs multiple GPUs (1.b.)

## 1.a. SINGLE GPU:
Start an experiment with the following choices:
- Profile: Open-Stack
- Hardware type: Cloudlab Wiscosin -> c240g5 (or another hardware with at least 1 GPU)
- Number of compute nodes (at Site 1) -> 0
- Schedule experiment for a time when the nodes are availabile 

Open terminal and ssh into node 

```bash
#EXAMPLE:
ssh vailg@c240g5-110219.wisc.cloudlab.us

```

## 1.b. MULTIPLE GPU: 

Start an experiment with the following choices:
- Profile: Open-Stack
- Hardware type: Cloudlab Wisconin -> c4130
- Number of compute nodes (at Site 1) -> 0
- Schedule experiment for a time when the nodes are available

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

After setting this up, you should start a new terminal window, and ssh into the node again and use the ```bash nvidia-smi ``` command to monitor the GPU usage while you run the tasks. 

## 4. Create a virtual enviornment to run the machine learning tasks

```bash
python3 -m venv tf

source tf/bin/activate

pip install --upgrade pip


pip install tensorflow[and-cuda]

#Verify setup
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#Check version of cuda complier
nvcc --version

```

If this doesn't work try this command: 

```bash

pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd

```

Run the check-gpu.py script to see if tensorflow and cude recognizes the GPU

```bash

python3 check_gpu.py

```
The output should say that there are one or multiple GPU(s) available depedning on the server you used. 


## 5. Run Scheduling Algorithms on ML tasks (includes both fundamental and reinforcement learning):

Go to /scheduling-algo/basic-scheduling-algos/ to find the files for the Python code for each baseline algorithm.

We have provided the code for all scheduling algorithms that have been used for this project. As long as the repository has been cloned, you should have all of the code available on the node. 

## Directory Overview

(1) FCFS_scheduling_v2.py
First come first serve scheduling. For this code, all the tasks are already implemented inside of the .py file. To change the task, mimic the same formatting used in the code. There are also other tasks found in the scheduling-algo/tasks/ directory that can be copied and pasted into this file. After changing that, you need to update the list of tasks and change it to the name of your task. There are comments in the code explaining how to do this.

TO RUN THIS CODE:

```python
python3 FCFS_scheduling_v2.py
```

(2) RL_Scheduling_v2.py
Reinforcement learning scheduling algorithm. This code requires the user to have all the tasks in their own indivdual files. For the tasks we used for this report, you can see they are in the same directory, and are called in the code by the name of their file. For the user to use their own deep learning tasks, or use another type from the task directory , just make sure to change the path to the tasks and update the name of the task in the list.

All of the following files are the tasks used for the scheduling algorithms and are further explained in the report and in the comments in each code:
(3) run_matrix_multiplication.py
(4) train_deep_learning.py
(5) train_lstm.py
(6) train_xgboost.py



---

# Simulating GPUs


We provide this option to simulate GPUs instead of using the actual GPUs on cloudlab. We consider this option because it was very diffcult to get access to the nodes with multiple GPUs. While we did achieve some results, there is still this option to consider. 


Ex: Evaluation function and task generator to simulate tasks:

### Task Generator:
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
