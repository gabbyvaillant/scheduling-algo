from collections import deque


class Process:
    def __init__(self, pid, burst_time):
        self.pid = pid          #Process ID
        self.burst_time = burst_time  #Time required to complete process
        self.remaining_time = burst_time  #Time left to finish process


def round_robin(processes, quantum):
    queue = deque(processes)  # Initialize the process queue
    time = 0  # Simulation clock
    
    while queue:
        process = queue.popleft()  # Get the next process from the queue
        print(f"Time {time}: Process {process.pid} starts execution")
        
        # Check if process can finish within its time slice
        if process.remaining_time > quantum:
            time += quantum
            process.remaining_time -= quantum
            print(f"Process {process.pid} did not finish. Remaining time: {process.remaining_time}")
            queue.append(process)  # Re-add the process to the queue
        else:
            time += process.remaining_time
            print(f"Process {process.pid} finished at time {time}")
            process.remaining_time = 0
    
    print(f"All processes completed by time {time}")

# Example usage
# Change the different lengths of processes
process_list = [
    Process(1, 11),  # Process 1 with burst time 10
    Process(2, 5),   # Process 2 with burst time 5
    Process(3, 8),   # Process 3 with burst time 8
]

# Ask the user for the quantum time
quantum = int(input("Enter the time slice (quantum) for each process: "))  # Get user input
round_robin(process_list, quantum)
