# this code expects mlflow ui to be running at 5000
# open a new cmd terminal in VS code, and run: mlflow ui --port 5000
# switch to a new terminal to execute this .py script after that using "python play_lin.....""
# this is used along with play_linreg_mlflow_streamlit_use_model_v2.py
# this script creates a model and logs in mlflow while the "use" script use that model
# mlflow serves the model.pkl in this case


import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [2, 3, 5, 8, 11]

model = LinearRegression()
model.fit(X, y)

# The tracking URI in MLflow is used to specify the location where MLflow should log the run information. 
# When you’re running your script in a Jupyter notebook, the tracking URI is automatically set to the CWD
# if you don’t specify it explicitly. 
# This is why your code works in a Jupyter notebook even without setting the tracking URI.
# It is a good practice to set it explicitly in py and ipynb.


# Set the tracking URI to the local tracking server
mlflow.set_tracking_uri('http://localhost:5000')

# Define the experiment name
experiment_name = "mlflow_streamlit_experiment"

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    # If the experiment does not exist, create it
    mlflow.create_experiment(experiment_name)
    
# Set the experiment
mlflow.set_experiment(experiment_name)

# Log model with MLflow
with mlflow.start_run(run_name="SLR_Model"):
    mlflow.sklearn.log_model(model, "model")
