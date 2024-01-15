# mlflow should be running at 5000
# the model and version must be visible as registered.

import numpy as np
import mlflow
import mlflow.pyfunc

# Set the tracking URI to the local tracking server
mlflow.set_tracking_uri('http://localhost:5000')

model_name = "play_SLR_model"
model_version = 1

model_uri = f"models:/{model_name}/{model_version}"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Define a test row
test_row = np.array([[6]])

# Use the loaded model to make a prediction on the test row
prediction = loaded_model.predict(test_row)

print(f"The predicted value for the test row {test_row} is {prediction}")