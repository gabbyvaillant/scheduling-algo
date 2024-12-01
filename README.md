# Optimizing GPU Scheduling for Machine Learning Workloads

This project explores scheduling algorithms for deep learning tasks, focusing on optimizing GPU utilization. We compare basic scheduling methods across four deep learning tasks under heavy GPU load, then introduce a reinforcement learning-based approach to improve GPU efficiency. Unlike traditional methods that focus on CPU usage or makespan, our approach prioritizes GPU usage, which is crucial as GPUs are increasingly used to accelerate deep learning. To address the lack of datasets with GPU usage information, we created a set of synthetic deep learning and machine learning tasks to test and refine the algorithms.


## Set up:

1. Set up experiment on Cloud Lab
This project uses Cloudlab for GPU access. Students in AMS 560 will have access through this link: https://www.cloudlab.us/show-project.php?project=AMS560-SBU

Check to see the resource availability: https://www.cloudlab.us/resinfo.php

Start an experiment with the following choices:
- Profile: Open-Stack
- Hardware type: Cloudlab Wiscosin -> d7525 (or another hardware with GPU access)
- Number of compute nodes (at Site 1) -> 1
- Schedule experiment for a time where there is resouce avilability 

2. Set up GPU usage monitor
To set up the GPU usage montior, use the commands in the script titled GPU-setup.sh once on Cloudlab

3. Run basline scheduling algorithms (FCFS, Round-Robin) on tasks

4. Run reinforcement learning based scheduling algorithms on tasks

Contributors: Gabrielle Vaillant, Michael Deisler, Iftekhar Alam
