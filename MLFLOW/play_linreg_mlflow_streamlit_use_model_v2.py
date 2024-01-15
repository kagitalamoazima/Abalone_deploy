# this is used along with play_linreg_mlflow_streamlit_fit_model.py
# this script expects an experiment with the given name to exist in mlflow mlruns and shows in UI
# it fetches the model.pkl from the latest run and uses in the streamlit webapp hosted locally
# execute this script using "streamlit run play_linreg_...py"


import streamlit as st
import mlflow.sklearn
import pandas as pd

# Set the tracking URI to the local tracking server
# mlflow.set_tracking_uri('http://localhost:5000')

# Define the experiment name
experiment_name = "mlflow_streamlit_experiment"

# Get the experiment
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    mlflow.create_experiment(experiment_name)
    print(f"Experiment '{experiment_name}' created successfully.")
else:
    print(f"Experiment '{experiment_name}' already exists.")

if experiment is not None:
    # Get all runs for the experiment
    runs = mlflow.search_runs(experiment_ids=experiment.experiment_id)
else:
    print("Experiment not found")
    runs = None

if runs is not None:
    # Sort the runs by start time and get the latest run
    latest_run = runs.sort_values("start_time", ascending=False).iloc[0]

    # Get the run ID of the latest run
    run_id = latest_run.run_id

    # Construct the model URI
    model_uri = f"runs:/{run_id}/model"

    # Load the model logged with MLflow as a scikit-learn model
    model = mlflow.sklearn.load_model(model_uri)
else:
    print("No runs found for the experiment")
    model = None

#Create the Streamlit web app
st.header("Streamlit demo")

st.sidebar.header("This is a web app")

X_test = st.sidebar.slider("Select X to get yhat", 0, 10, 5)

st.write("X test is:", X_test)

if model is not None:
    yhat_test = model.predict([[X_test]])

    st.write("b0 is", round(model.intercept_, 3))
    st.write("b1 is", round(model.coef_[0], 3))
    st.write("yhat test is", yhat_test)
else:
    st.write("Model not found")
