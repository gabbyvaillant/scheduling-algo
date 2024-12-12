import heapq
import matplotlib.pyplot as plt
import random

# Define the task durations and GPU usage (in arbitrary time units and GPU units)
tasks = [(random.randint(5, 15), random.uniform(30, 80)) for _ in range(10)]
# Each task is now a tuple: (duration, gpu_usage)

# Number of GPUs
gpus = 4

# Define GPU capacities
gpu_capacities = {i: 100 for i in range(gpus)}  # Each GPU has a capacity of 100

# Initialize a min-heap to track GPU workloads (total workload, gpu_index)
gpu_heap = [(0, i) for i in range(gpus)]

# List to track task assignments
task_assignments = []

time_usage = {i: [] for i in range(gpus)}  # Track time usage for each GPU

# Simulate task scheduling
for duration, gpu_usage in tasks:
    # Find a GPU that can handle the task's GPU usage requirement
    available_gpus = [item for item in gpu_heap if gpu_capacities[item[1]] >= gpu_usage]

    if not available_gpus:
        raise ValueError("No GPU can handle the task's GPU usage requirement")

    # Get the least loaded GPU from the available ones
    least_loaded_workload, gpu_index = min(available_gpus, key=lambda x: x[0])
    gpu_heap.remove((least_loaded_workload, gpu_index))

    # Assign the task to this GPU
    start_time = least_loaded_workload
    end_time = start_time + duration
    task_assignments.append((duration, gpu_usage, gpu_index, start_time, end_time))

    # Track usage over time
    time_usage[gpu_index].append((start_time, end_time, gpu_usage))

    # Update the workload and push it back into the heap
    new_workload = end_time
    heapq.heappush(gpu_heap, (new_workload, gpu_index))

# Plot Gantt Chart
gantt_data = [(gpu, start, end, duration, gpu_usage) for duration, gpu_usage, gpu, start, end in task_assignments]
fig, ax = plt.subplots(figsize=(10, 6))
for gpu, start_time, end_time, duration, gpu_usage in gantt_data:
    ax.barh(gpu, end_time - start_time, left=start_time, edgecolor="black", label=f"Task {duration}s, {gpu_usage:.1f} GPU")
ax.set_yticks(range(gpus))
ax.set_yticklabels([f"GPU {i}" for i in range(gpus)])
ax.set_xlabel("Time")
ax.set_title("Gantt Chart of Task Scheduling with GPU Usage")
ax.grid(axis="x", linestyle="--", alpha=0.7)
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), title="Task Details")
plt.tight_layout()
plt.show()

# Plot GPU Usage Over Time
usage_changes = []
for gpu, intervals in time_usage.items():
    for start, end, usage in intervals:
        usage_changes.append((start, usage, gpu, 'start'))
        usage_changes.append((end, -usage, gpu, 'end'))

# Sort changes by time
usage_changes.sort()

# Calculate GPU usage over time for each GPU
gpu_usage_timeline = {i: [] for i in range(gpus)}
current_time = 0
current_usage = {i: 0 for i in range(gpus)}

for time, usage, gpu, event in usage_changes:
    if time != current_time:
        for i in range(gpus):
            gpu_usage_timeline[i].append((current_time, current_usage[i]))
        current_time = time
    current_usage[gpu] += usage

for i in range(gpus):
    gpu_usage_timeline[i].append((current_time, current_usage[i]))

# Plot the line chart for each GPU
fig, ax = plt.subplots(figsize=(10, 6))
for gpu, timeline in gpu_usage_timeline.items():
    times = [t[0] for t in timeline]
    usage_values = [t[1] for t in timeline]
    ax.step(times, usage_values, where='post', label=f"GPU {gpu}", linewidth=2)
ax.set_xlabel("Time")
ax.set_ylabel("GPU Usage")
ax.set_title("GPU Usage Over Time (Dynamic Resource Scheduling)")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
