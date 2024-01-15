#Track the model using mlflow

#call the pickel model
import mlflow
import pickle

with open("Deployment_model.pkl","rb") as file:
    load_model = pickle.load(file)