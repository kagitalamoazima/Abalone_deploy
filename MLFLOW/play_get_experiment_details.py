import mlflow.pyfunc
import pandas as pd

# Define the experiment name
experiment_name = "my_experiment"

# Get the experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
print(experiment)