import matplotlib.pyplot as plt
import numpy as np
import random

class Job:
    def __init__(self, job_name, total_time, avg_gpu_usage, avg_gpu_memory):
        self.job_name = job_name
        self.total_time = total_time
        self.remaining_time = total_time
        self.avg_gpu_usage = avg_gpu_usage
        self.avg_gpu_memory = avg_gpu_memory

    def run(self):
        executed_time = self.remaining_time
        self.remaining_time = 0
        return executed_time

class MultiGPUFCFSScheduler:
    def __init__(self, num_gpus=4):
        self.num_gpus = num_gpus
        self.job_queue = []
        self.execution_log = []  # To track the execution for Gantt chart
        self.gpu_logs = {i: [] for i in range(self.num_gpus)}  # Separate logs for each GPU

    def add_job(self, job):
        self.job_queue.append(job)

    def run(self):
        current_time = 0
        gpu_usage = [0] * self.num_gpus  # Track GPU usage for each GPU
        gpu_memory = [0] * self.num_gpus  # Track GPU memory for each GPU
        job_assignments = []  # To track job allocation to each GPU for Gantt chart

        # First-Come, First-Serve scheduling
        while self.job_queue:
            for gpu_id in range(self.num_gpus):
                if not self.job_queue:
                    break
                current_job = self.job_queue.pop(0)
                start_time = current_time

                # Execute the job completely
                executed_time = current_job.run()
                end_time = start_time + executed_time

                # Log job assignment and execution
                job_assignments.append((current_job.job_name, gpu_id, start_time, end_time))
                self.execution_log.append((current_job.job_name, start_time, end_time))

                # Simulate GPU usage and memory usage for the job
                time_increment = 0.1  # Log GPU usage every 0.1 seconds
                t = start_time
                while t < end_time:
                    gpu_usage[gpu_id] = self._smooth_transition(gpu_usage[gpu_id], current_job.avg_gpu_usage)
                    gpu_memory[gpu_id] = self._smooth_transition(gpu_memory[gpu_id], current_job.avg_gpu_memory)

                    self.gpu_logs[gpu_id].append((t, gpu_usage[gpu_id], gpu_memory[gpu_id]))
                    t += time_increment

                current_time = end_time

        self.job_assignments = job_assignments  # Save job assignments for plotting

    def _smooth_transition(self, current_value, target_value):
        # Smooth transition with reduced noise
        noise = np.random.normal(0, 2)  # Smaller noise added to the current value
        transition_speed = 0.02  # Slower transition speed
        new_value = current_value + (target_value - current_value) * transition_speed + noise
        return max(0, min(100, new_value))  # Clamp between 0 and 100%

    def plot_gantt_chart(self):
        # Create a list of unique job names for color coding
        job_colors = {job_name: plt.cm.tab20(i / len(set(job_name for job_name, _, _, _ in self.job_assignments)))
                      for i, job_name in enumerate(set(job_name for job_name, _, _, _ in self.job_assignments))}

        # Plot Gantt Chart showing each job's allocation to specific GPU
        plt.figure(figsize=(10, 6))
        for job_name, gpu_id, start_time, end_time in self.job_assignments:
            color = job_colors[job_name]
            plt.barh(gpu_id, end_time - start_time, left=start_time, edgecolor='black', color=color, label=job_name if gpu_id == 0 else "")

        # Set GPU labels on the y-axis
        plt.yticks(range(self.num_gpus), [f'gpu-{i}' for i in range(self.num_gpus)])
        plt.xlabel('Time (s)')
        plt.ylabel('GPUs')
        plt.title('First-Come, First-Serve Scheduling Gantt Chart (Job Allocation per GPU)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Avoid duplicate labels in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))  # Remove duplicates
        plt.legend(unique_labels.values(), unique_labels.keys())
        plt.show()

    def plot_gpu_usage_and_memory(self):
        # Plot GPU Usage and Memory Usage for all GPUs
        plt.figure(figsize=(10, 6))

        for gpu_id in range(self.num_gpus):
            times, gpu_usages, gpu_memories = zip(*self.gpu_logs[gpu_id])

            # Scale the GPU memory usage to be similar but lower than GPU usage
            gpu_memories_scaled = [mem * 0.7 for mem in gpu_usages]  # Scaling memory to 70% of GPU usage

            # Plot GPU Usage and Memory Usage
            plt.plot(times, gpu_usages, label=f'GPU {gpu_id} Usage (%)', linewidth=2)

        plt.xlabel('Time (s)')
        plt.ylabel('Usage (%)')
        plt.ylim(0, 100)
        plt.title('GPU Usage Over Time (Multi-GPU Setup)')
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

# Example usage
scheduler = MultiGPUFCFSScheduler(num_gpus=4)

# Adding 15 random jobs with random parameters
for i in range(15):
    total_time = random.uniform(5, 50)  # Job execution time between 5 and 50 seconds
    avg_gpu_usage = random.uniform(20, 80)  # Average GPU usage between 20% and 80%
    avg_gpu_memory = random.uniform(10, 70)  # Average GPU memory usage between 10% and 70%
    scheduler.add_job(Job(f"Job{i+1}", total_time, avg_gpu_usage, avg_gpu_memory))

scheduler.run()
scheduler.plot_gantt_chart()
scheduler.plot_gpu_usage_and_memory()
