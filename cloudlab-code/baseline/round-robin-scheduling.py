import os
import subprocess
import time
import pynvml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Initialize NVIDIA Management Library
pynvml.nvmlInit()

# Task Directory and Task List
task_directory = "/scheduling-algo/tasks"
task_files = [
    "CIFAR-IC.py",
    "MNIST-IC.py",
    "FMNIST-IC.py",
    "IMAGENET-IC.py",
    "IMBD-LSTM.py",
    "STOCK-LSTM.py",
    "SHAKE-LSTM.py",
    "BERT-TRANSF.py",
    "Reuters-TRANSF.py",
    "GPT2-TRANSF.py",
    "TRANSLATION-TRANSF.py"
]

# Construct full paths
task_files = [os.path.join(task_directory, task) for task in task_files]

def get_gpu_usage():
    """Returns the current GPU usage percentage using nvidia-smi command"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Split and convert to list of integers
        gpu_utilization = [int(util.strip()) for util in result.stdout.strip().split('\n')]
        return gpu_utilization[0]  # Since only one GPU, return the usage for the first GPU
    except Exception as e:
        print(f"Error retrieving GPU usage: {e}")
        return -1

def round_robin(task_files, time_quantum=10, total_runtime=120):
    """
    Round Robin Scheduler for GPU-aware task allocation for a single GPU
    
    Inputs:
    task_files: List of task files
    time_quantum: Time slice for each task (seconds)
    total_runtime: Total runtime of the scheduler (seconds)
    
    Output: 
    Graphs of GPU usage, and chart of GPU usage, and list of how the tasks were arranged
    """
    # Tracking variables
    gpu_usage_history = []
    task_allocation_log = []
    current_time = 0
    num_tasks = len(task_files)
    task_index = 0
    
    print("Starting Round Robin Scheduling for a Single GPU...")
    
    while current_time < total_runtime:
        # Get current task file
        current_task = task_files[task_index]
        
        # Get current GPU usage
        gpu_usage = get_gpu_usage()
        
        # Log task allocation
        task_allocation_log.append({
            'time': current_time,
            'task': os.path.basename(current_task),
            'gpu': 0  # Only one GPU, so set GPU index to 0
        })
        
        print(f"\n--- Scheduling Task: {current_task}")
        print(f"Current GPU Usage: {gpu_usage}%")
        
        # Set environment to target the single GPU (GPU 0)
        task_env = os.environ.copy()
        task_env['CUDA_VISIBLE_DEVICES'] = "0"
        
        try:
            # Start the task
            process = subprocess.Popen(
                ["python", current_task], 
                env=task_env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # Monitor task
            task_start_time = time.time()
            task_runtime = 0
            while task_runtime < time_quantum and current_time < total_runtime:
                # Get and log GPU usage
                current_gpu_usage = get_gpu_usage()
                gpu_usage_history.append({
                    'time': current_time,
                    'gpu_usage': current_gpu_usage
                })
                
                # Check if process is still running
                if process.poll() is not None:
                    print(f"Task {current_task} completed early")
                    break
                
                # Sleep and update times
                time.sleep(1)
                task_runtime = time.time() - task_start_time
                current_time += 1
            
            # Terminate if still running
            if process.poll() is None:
                print(f"Terminating Task: {current_task} after {time_quantum} seconds")
                process.terminate()
                # Give a moment to terminate gracefully
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
            
            # Capture and log output/errors
            stdout, stderr = process.communicate()
            if stdout:
                print("Task STDOUT:", stdout.decode())
            if stderr:
                print("Task STDERR:", stderr.decode())
        
        except Exception as e:
            print(f"Error running task {current_task}: {e}")
        
        # Move to next task in round-robin fashion
        task_index = (task_index + 1) % num_tasks
    
    # Create visualizations
    _create_gpu_usage_visualization(gpu_usage_history, task_allocation_log)
    
    return gpu_usage_history, task_allocation_log

def _create_gpu_usage_visualization(gpu_usage_history, task_allocation_log):
    """
    Create visualizations for GPU usage and task allocation for a single GPU
    """
    # Prepare data for plotting
    times = [entry['time'] for entry in gpu_usage_history]
    gpu_usages = [entry['gpu_usage'] for entry in gpu_usage_history]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot GPU Usage over time
    ax1.plot(times, gpu_usages, label='GPU Usage (%)', color='blue')
    ax1.set_title('GPU Utilization Over Time')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('GPU Utilization (%)')
    ax1.legend()
    
    # Plot Task Allocation Timeline
    task_names = [entry['task'] for entry in task_allocation_log]
    times = [entry['time'] for entry in task_allocation_log]
    
    scatter = ax2.scatter(times, [0] * len(times), c=range(len(times)), 
                          cmap='viridis', s=100)
    ax2.set_title('Task Allocation Timeline')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('GPU Index (0 for single GPU)')
    
    # Add task name annotations
    for i, (x, task) in enumerate(zip(times, task_names)):
        ax2.annotate(task, (x, 0), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8, 
                     color='black', rotation=45)
    
    plt.colorbar(scatter, ax=ax2, label='Task Sequence')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('gpu_scheduling_analysis.png')
    plt.close()
    
    # Create a detailed CSV log
    df = pd.DataFrame(task_allocation_log)
    df.to_csv('task_allocation_log.csv', index=False)
    
    print("Visualization saved as 'gpu_scheduling_analysis.png'")
    print("Task allocation log saved as 'task_allocation_log.csv'")

# Run the scheduler
try:
    # You can adjust these parameters as needed
    gpu_usage, task_log = round_robin(task_files, time_quantum=10, total_runtime=120)
except KeyboardInterrupt:
    print("Scheduler stopped.")
finally:
    # Cleanup NVIDIA Management Library
    pynvml.nvmlShutdown()
