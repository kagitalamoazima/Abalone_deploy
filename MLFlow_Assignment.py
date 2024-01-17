import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

experiment_name = "Regression_Model_Performance"

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    # If the experiment does not exist, create it
    mlflow.create_experiment(experiment_name)
    
# Set the experiment
mlflow.set_experiment(experiment_name)



df = pd.read_csv("abalonedata.csv")

df_num=df.select_dtypes(include='number')
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 -Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3+1.5*IQR
df[df_num.columns] = df_num.clip(lower=lower_bound, upper=upper_bound, axis=1)


X=df.drop('Age',axis=1)
y=df.Age


from sklearn.impute import SimpleImputer
steps = [('imputation_mean', SimpleImputer(missing_values=np.nan, strategy= 'mean')), ('scaler', MinMaxScaler())]

from sklearn.pipeline import Pipeline

numeric_processor  = Pipeline(steps)


from sklearn.preprocessing import OneHotEncoder

steps_cat = [('imputation_constant', SimpleImputer(fill_value='missing', strategy='constant')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]

cat_processor = Pipeline(steps_cat)





from sklearn.compose import ColumnTransformer

preprocessors = ColumnTransformer(
    [('categorical', cat_processor,['Sex']),
    ('numerical', numeric_processor, ['Length','Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Rings'])]
    

)

from sklearn.pipeline import make_pipeline

Linear_pipe = make_pipeline(preprocessors, LinearRegression())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

Linear_pipe.fit(X_train, y_train)

linear_y_pred = Linear_pipe.predict(X_test)
linear_y_train_pred = Linear_pipe.predict(X_train)



decision_pipe = make_pipeline(preprocessors, DecisionTreeRegressor(max_depth=3))

decision_pipe.fit(X_train, y_train)

decision_y_pred = decision_pipe.predict(X_test)
decision_y_train_pred = decision_pipe.predict(X_train)



ada_boost_pipe = make_pipeline(preprocessors, AdaBoostRegressor(base_estimator= DecisionTreeRegressor(max_depth=3), n_estimators=50, learning_rate=1.0, random_state=42))

ada_boost_pipe.fit(X_train, y_train)

ada_y_pred = ada_boost_pipe.predict(X_test)
ada_y_train_pred = ada_boost_pipe.predict(X_train)




with mlflow.start_run(run_name = 'Linear_Model_run'):
    # Log parameters
    mlflow.log_param("model", "Linear Regression")
    
    # Log metrics
    linear_mse = mean_squared_error(y_test, linear_y_pred)
    linear_mse_train = mean_squared_error(y_train, linear_y_train_pred)
    mlflow.log_metric("linear_mse", linear_mse)
    mlflow.log_metric("linear_train_mse", linear_mse_train)
    # Log the Linear Regression model
    mlflow.sklearn.log_model(Linear_pipe, "linear_model")


with mlflow.start_run(run_name= 'Decision_Model_run'):
    # Log parameters
    mlflow.log_param("model", "DecisionTree Regressor")
    
    # Log metrics
    dec_mse = mean_squared_error(y_test, decision_y_pred)
    dec_mse_train = mean_squared_error(y_train, decision_y_train_pred)
    mlflow.log_metric("decision_mse", dec_mse)
    mlflow.log_metric("decision_train_mse", dec_mse_train)
    # Log the Linear Regression model
    mlflow.sklearn.log_model(decision_pipe, "DecisionTree_model")


with mlflow.start_run(run_name = 'Ada_Boost_Model_run'):
    # Log parameters
    mlflow.log_param("model", "ADA Boost Regressor")
    
    # Log metrics
    ada_mse = mean_squared_error(y_test, ada_y_pred)
    ada_mse_train = mean_squared_error(y_train, ada_y_train_pred)
    mlflow.log_metric("ada_mse", ada_mse)
    mlflow.log_metric("ada_train_mse", ada_mse_train)
    # Log the Linear Regression model
    mlflow.sklearn.log_model(ada_boost_pipe, "ADA_Boost_model")