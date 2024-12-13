import matplotlib.pyplot as plt
import numpy as np

class Job:
    def __init__(self, job_name, total_time, avg_gpu_usage, avg_gpu_memory):
        self.job_name = job_name
        self.total_time = total_time
        self.remaining_time = total_time
        self.avg_gpu_usage = avg_gpu_usage
        self.avg_gpu_memory = avg_gpu_memory

    def run(self, time_slice):
        executed_time = min(time_slice, self.remaining_time)
        self.remaining_time -= executed_time
        return self.remaining_time == 0

class MultiGPURFCFScheduler:
    def __init__(self, time_slice, num_gpus=3):
        self.time_slice = time_slice
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

        # Round Robin scheduling: Process jobs in the order they are added to the queue
        while self.job_queue:
            current_job = self.job_queue.pop(0)
            start_time = current_time
            executed_time = min(self.time_slice, current_job.remaining_time)

            # Simulate GPU selection for job allocation (first available GPU)
            selected_gpu = int(current_time % self.num_gpus)  # Ensure it's an integer index

            # Assign job to GPU and log this for the Gantt chart
            job_assignments.append((current_job.job_name, selected_gpu, start_time, start_time + executed_time))

            # Run the job
            if current_job.run(self.time_slice):
                print(f"Job {current_job.job_name} completed at time {current_time + executed_time:.2f}s.")
            else:
                print(f"Job {current_job.job_name} paused at time {current_time + executed_time:.2f}s.")
                self.job_queue.append(current_job)

            # Log the execution for the Gantt chart
            self.execution_log.append((current_job.job_name, start_time, start_time + executed_time))

            # Simulate GPU usage and memory usage with smooth transition for each GPU
            time_increment = 0.1  # Log GPU usage every 0.1 seconds
            t = start_time
            while t < start_time + executed_time:
                # Round time `t` to the nearest integer for indexing
                t_rounded = round(t)

                # Smooth transition for GPU usage and memory
                gpu_usage[selected_gpu] = self._smooth_transition(gpu_usage[selected_gpu], current_job.avg_gpu_usage)
                gpu_memory[selected_gpu] = self._smooth_transition(gpu_memory[selected_gpu], current_job.avg_gpu_memory)

                # Log the GPU usage and memory at each time step
                self.gpu_logs[selected_gpu].append((t_rounded, gpu_usage[selected_gpu], gpu_memory[selected_gpu]))

                t += time_increment  # Increment time by time_increment

            current_time += executed_time

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
            # Assign a color to the job based on the job name
            color = job_colors[job_name]
            plt.barh(gpu_id, end_time - start_time, left=start_time, edgecolor='black', color=color, label=job_name if gpu_id == 0 else "")

        # Set GPU labels on the y-axis
        plt.yticks(range(self.num_gpus), [f'gpu-{i}' for i in range(self.num_gpus)])

        plt.xlabel('Time (s)')
        plt.ylabel('GPUs')
        plt.title('Round Robin Scheduling Gantt Chart (Job Allocation per GPU)')
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
time_slice = 10  # Time slice in seconds
scheduler = MultiGPURFCFScheduler(time_slice, num_gpus=3)

# Adding jobs with real running times and average GPU usage and memory usage
scheduler.add_job(Job("XGBoost_job1", 2.77, 30, 40))       # Job name, total time, avg GPU usage, avg GPU memory usage
scheduler.add_job(Job("DL_job1", 80.83, 70, 60))
scheduler.add_job(Job("MatrixMultiplication_job1", 7.24, 40, 50))
scheduler.add_job(Job("LSTM_job1", 34.18, 60, 50))

scheduler.run()
scheduler.plot_gantt_chart()
scheduler.plot_gpu_usage_and_memory()
