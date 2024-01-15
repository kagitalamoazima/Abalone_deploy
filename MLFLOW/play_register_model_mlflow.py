import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [2, 3, 5, 8, 11]

model = LinearRegression()
model.fit(X, y)

# Set the tracking URI to the local tracking server
mlflow.set_tracking_uri('http://localhost:5000')

# Define the experiment name
experiment_name = "my_experiment"

# Set the experiment
mlflow.set_experiment(experiment_name)

# Log model with MLflow
with mlflow.start_run(run_name="SLR_Model"):
    mlflow.sklearn.log_model(model, "model")
    run_id = mlflow.active_run().info.run_id
    artifact_uri = "runs:/" + run_id + "/model"

# Register the model
model_details = mlflow.register_model(
    model_uri=artifact_uri,
    name="play_SLR_model"
)

