import time
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class XGBoostJob:
    def __init__(self, n_estimators=100, max_depth=6, job_name="XGBoostJob"):
        """
        Initialize the XGBoost job with specific parameters.
        :param n_estimators: Number of boosting rounds.
        :param max_depth: Maximum depth of each tree.
        :param job_name: Name of the job for identification.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.job_name = job_name

    def get_command(self):
        return f"python train_xgboost.py --n_estimators {self.n_estimators} --max_depth {self.max_depth}"

xgJob = XGBoostJob(n_estimators=1200, max_depth=30, job_name="XGBoost_Job1")
# Train an XGBoost model on synthetic data and measure the time taken.
print(f"Job {xgJob.job_name} started.")

# Generate synthetic dataset
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix and load onto the GPU
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up XGBoost parameters for GPU usage
params = {
    "objective": "binary:logistic",
    "max_depth": xgJob.max_depth,
    "tree_method": "hist",
    "device": "cuda"
}

# Measure time taken for training
start_time = time.time()
model = xgb.train(params, dtrain, num_boost_round=xgJob.n_estimators)
end_time = time.time()
# Test model accuracy
predictions = (model.predict(dtest) > 0.5).astype(int)
accuracy = accuracy_score(y_test, predictions)
print(f"Job {xgJob.job_name} completed. Time taken: {end_time - start_time} seconds.")
print(f"Accuracy of {xgJob.job_name}: {accuracy}")