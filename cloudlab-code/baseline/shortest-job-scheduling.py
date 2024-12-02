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
task_directory = "/scheduling-algo/tasks" #change path
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

def measure_task_time(task_file, warm_up_time=2):
    """
    Measures the execution time of a task by running it briefly to estimate its runtime.
    
    Parameters:
    - task_file: Path to the Python task file.
    - warm_up_time: Time (in seconds) to run the task for measuring the time.
    
    Returns:
    - Estimated runtime of the task.
    """
    try:
        task_env = os.environ.copy()
        task_env['CUDA_VISIBLE_DEVICES'] = "0"
        
        # Start the task and measure execution time
        start_time = time.time()
        process = subprocess.Popen(
            ["python", task_file],
            env=task_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Allow the task to run for a warm-up period (just a few seconds)
        time.sleep(warm_up_time)
        
        # Terminate the process after warm-up period
        process.terminate()
        process.wait()
        
        end_time = time.time()
        
        # Measure the time taken to complete the warm-up
        execution_time = end_time - start_time
        print(f"Task {task_file} execution time (approximate): {execution_time:.2f} seconds")
        
        return execution_time
    except Exception as e:
        print(f"Error measuring task time for {task_file}: {e}")
        return float('inf')  # Return a high value if measurement fails

def shortest_job_first(task_files, total_runtime=120):
    """
    Shortest Job First (SJF) Scheduler for GPU-aware task allocation for a single GPU
    
    Inputs:
    task_files: List of task files
    total_runtime: Total runtime of the scheduler (seconds)
    
    Output: 
    Graphs of GPU usage, and chart of GPU usage, and list of how the tasks were arranged
    """
    # Measure the run time of each task
    task_execution_times = {}
    for task_file in task_files:
        execution_time = measure_task_time(task_file)
        task_execution_times[task_file] = execution_time
    
    # Sort the tasks by their execution time (Shortest Job First)
    sorted_task_files = sorted(task_files, key=lambda x: task_execution_times[x])

    # Tracking variables
    gpu_usage_history = []
    task_allocation_log = []
    current_time = 0
    num_tasks = len(sorted_task_files)
    
    print("Starting Shortest Job First Scheduling for a Single GPU...")
    
    for task_index in range(num_tasks):
        # Get current task file
        current_task = sorted_task_files[task_index]
        
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
            while current_time < total_runtime:
                # Get and log GPU usage
                current_gpu_usage = get_gpu_usage()
                gpu_usage_history.append({
                    'time': current_time,
                    'gpu_usage': current_gpu_usage
                })
                
                # Check if process is still running
                if process.poll() is not None:
                    print(f"Task {current_task} completed")
                    break
                
                # Sleep and update times
                time.sleep(1)
                current_time += 1
            
            # Capture and log output/errors
            stdout, stderr = process.communicate()
            if stdout:
                print("Task STDOUT:", stdout.decode())
            if stderr:
                print("Task STDERR:", stderr.decode())
        
        except Exception as e:
            print(f"Error running task {current_task}: {e}")
    
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
    plt.savefig('gpu_scheduling_analysis_sjf.png')
    plt.close()
    
    # Create a detailed CSV log
    df = pd.DataFrame(task_allocation_log)
    df.to_csv('task_allocation_log_sjf.csv', index=False)
    
    print("Visualization saved as 'gpu_scheduling_analysis_sjf.png'")
    print("Task allocation log saved as 'task_allocation_log_sjf.csv'")

# Run the scheduler
try:
    # You can adjust these parameters as needed
    gpu_usage, task_log = shortest_job_first(task_files, total_runtime=120)
except KeyboardInterrupt:
    print("Scheduler stopped.")
finally:
    # Cleanup NVIDIA Management Library
    pynvml.nvmlShutdown()
