# Optimizing GPU Scheduling for Machine Learning Workloads

This project explores scheduling algorithms for deep learning tasks, focusing on optimizing GPU utilization. We compare basic scheduling methods across four deep learning tasks under heavy GPU load, then introduce a reinforcement learning-based approach to improve GPU efficiency. Unlike traditional methods that focus on CPU usage or makespan, our approach prioritizes GPU usage, which is crucial as GPUs are increasingly used to accelerate deep learning. To address the lack of datasets with GPU usage information, we created a set of synthetic deep learning and machine learning tasks to test and refine the algorithms.


## 1. Set up experiment on Cloud Lab
This project uses Cloudlab for GPU access. Students in AMS 560 will have access through this link: https://www.cloudlab.us/show-project.php?project=AMS560-SBU

Check to see the resource availability: https://www.cloudlab.us/resinfo.php

Check to see which hardwares have access to GPUs: https://docs.cloudlab.us/hardware.html

Start an experiment with the following choices:
- Profile: Open-Stack
- Hardware type: Cloudlab Wiscosin -> d7525 (or another hardware with GPU access)
- Number of compute nodes (at Site 1) -> 0 or 1
- Schedule experiment for a time where there is resource availability 

Open terminal and ssh into node 

```bash
#EXAMPLE:
ssh vailg@c240g5-110219.wisc.cloudlab.us


```

## 2. Set up GPU driver

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


## 3. Create a virtual enviornment to run the machine learning tasks

Install miniconda to create a virtual enviornment on the node 
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda init --all

```
May need to reboot the system again 

```bash
conda create --name tf_gpu python=3.8

conda activate tf_gpu

conda install pip

pip install tensorflow==2.7

conda install -c conda-forge cudatoolkit=11.5 cudnn=8.3

```
Create a new python script to see if Tensorflow/cuda is recognizing the GPUs


```bash

nano check_gpu.py

```

In the python file enter the following code: 

```python

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

```
To save and close use hit Shift+control X and hit enter to save

Run the file

```bash

python3 check_gpu.py

```
The output should say that there are 1 GPU available. 

To set up the GPU usage montior, use the commands in the script titled GPU-setup.sh once on Cloudlab

4. Run basline scheduling algorithms (FCFS, Round-Robin) on tasks

5. Run reinforcement learning based scheduling algorithms on tasks

Contributors: Gabrielle Vaillant, Michael Deisler, Iftekhar Alam
