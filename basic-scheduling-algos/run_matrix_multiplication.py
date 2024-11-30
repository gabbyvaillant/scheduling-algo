import time
import tensorflow as tf
class MatrixMultiplicationJob:
    def __init__(self, matrix_size=5000, job_name="MatrixMultiplication"):
        """
        Initialize the matrix multiplication job with a specific matrix size.
        :param matrix_size: The size of the square matrices to multiply.
        :param job_name: Name of the job for identification.
        """
        self.matrix_size = matrix_size
        self.job_name = job_name
    def get_command(self):
        return f"python run_matrix_multiplication.py --matrix_size {self.matrix_size}"
    
mmJob = MatrixMultiplicationJob(matrix_size=30000, job_name="MatrixMultiplication_Job1")
    
# Perform matrix multiplication and measure the time taken.
print(f"Job {mmJob.job_name} started.")
        
# Generate two random matrices and perform matrix multiplication
start_time = time.time()
with tf.device('/GPU:0'):
    A = tf.random.uniform((mmJob.matrix_size, mmJob.matrix_size))
    B = tf.random.uniform((mmJob.matrix_size, mmJob.matrix_size))
    result = tf.matmul(A, B)
        
# measure the time
end_time = time.time()
        
print(f"Job {mmJob.job_name} completed. Time taken: {end_time - start_time} seconds.")