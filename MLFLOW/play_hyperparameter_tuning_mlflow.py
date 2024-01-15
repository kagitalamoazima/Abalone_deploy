import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the tracking URI to the local tracking server
mlflow.set_tracking_uri('http://localhost:5000')

# Define the experiment name
experiment_name = "Decision_Tree_Classifier"

# Set the experiment
mlflow.set_experiment(experiment_name)

# List of hyperparameters to tune
max_depth_values = [1, 2, 3, 4, 5]

for max_depth in max_depth_values:
    # Train the model
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log model with MLflow
    with mlflow.start_run(run_name=f"max_depth_{max_depth}"):
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
