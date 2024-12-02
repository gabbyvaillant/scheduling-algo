# Optimizing GPU Scheduling for Machine Learning Workloads

This project explores scheduling algorithms for deep learning tasks, focusing on optimizing GPU utilization. We compare basic scheduling methods across four deep learning tasks under heavy GPU load, then introduce a reinforcement learning-based approach to improve GPU efficiency. Unlike traditional methods that focus on CPU usage or makespan, our approach prioritizes GPU usage, which is crucial as GPUs are increasingly used to accelerate deep learning. To address the lack of datasets with GPU usage information, we created a set of synthetic deep learning and machine learning tasks to test and refine the algorithms.


## Set up:

1. Set up experiment on Cloud Lab
This project uses Cloudlab for GPU access. Students in AMS 560 will have access through this link: https://www.cloudlab.us/show-project.php?project=AMS560-SBU

Check to see the resource availability: https://www.cloudlab.us/resinfo.php

Check to see which hardwares have access to GPUs: https://docs.cloudlab.us/hardware.html

Start an experiment with the following choices:
- Profile: Open-Stack
- Hardware type: Cloudlab Wiscosin -> d7525 (or another hardware with GPU access)
- Number of compute nodes (at Site 1) -> 0 or 1
- Schedule experiment for a time where there is resouce avilability 

2. Set up GPU driver
```bash
sudo apt install ubuntu-drivers-common

sudo ubuntu-drivers list
# check if ubuntu-drivers install successfully

sudo ubuntu-drivers install

sudo reboot
# reboot the system after ubuntu-drivers install
# you will loss connection and this step need some time

# After reboot, reconnect with your system and check if NVIDIA drivers install successfully or not
nvidia-smi

# if error: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running. Try to repeat the above steps
```
To set up the GPU usage montior, use the commands in the script titled GPU-setup.sh once on Cloudlab

4. Run basline scheduling algorithms (FCFS, Round-Robin) on tasks

5. Run reinforcement learning based scheduling algorithms on tasks

Contributors: Gabrielle Vaillant, Michael Deisler, Iftekhar Alam
