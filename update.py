import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("abalonedata.csv")


# Streamlit app
st.title("Interactive Data Visualization")

st.set_option('deprecation.showPyplotGlobalUse', False)

# Dropdown for selecting plot type
plot_type = st.sidebar.selectbox("Select Plot Type", ["Histogram", "Boxplot", "Scatter Plot", "Pie Chart"])

# Dropdowns for selecting columns
x_column = st.sidebar.selectbox("Select X-axis Column", df.select_dtypes(include=['float64']).columns)


# Scatter plot-specific dropdown for hue (categorical variable)
hue_column = ""
if plot_type == "Scatter Plot":
    hue_column = st.sidebar.selectbox("Select Hue Column (Categorical)", df.select_dtypes(include=['object']).columns, index=0)
    y_column = st.sidebar.selectbox("Select Y-axis Column", df.select_dtypes(include=['float64']).columns)
# Plot based on user selection
if plot_type == "Histogram":
    st.subheader(f"Histogram for {x_column}")
    plt.hist(df[x_column], bins=20)
    st.pyplot()

elif plot_type == "Boxplot":
    st.subheader(f"Boxplot for {x_column}")
    sns.boxplot(x=x_column, data=df)
    st.pyplot()

elif plot_type == "Scatter Plot":
    st.subheader(f"Scatter Plot for {x_column} vs {y_column} with Hue: {hue_column}")
    sns.scatterplot(x=x_column, y=y_column, hue=hue_column, data=df)
    st.pyplot()

elif plot_type == "Pie Chart":
    st.subheader(f"Pie Chart for {x_column}")
    values = df[x_column].value_counts()
    st.pyplot(plt.pie(values, labels=values.index, autopct='%1.1f%%'))

df_num=df.select_dtypes(include='number')
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 -Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3+1.5*IQR
df_num = df_num.clip(lower=lower_bound, upper=upper_bound, axis=1)

x=df.drop('Age',axis=1) #Seperate fetures and target variable.
y=df.Age

x_num=x.select_dtypes('number')

scaler=MinMaxScaler()
x_num_scaled=scaler.fit_transform(x_num)

x_num_scaled=pd.DataFrame(x_num_scaled,columns=x_num.columns,index=x_num.index)

x_cat=x.select_dtypes('object')

x_cat_encoded=pd.get_dummies(x_cat,drop_first=True,dtype=int)

x = pd.concat([x_num_scaled, x_cat_encoded], axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


x.to_csv('Train_features_abalone.csv', index=False)
y.to_csv('predict_target_abalone.csv',index=False)

st.title("Regression Model Comparison")

selected_model = st.sidebar.selectbox("Select Regression Model", ["Decision Tree", "Linear Regression", "AdaBoost"])

Hyper_parameter_tuning = None

if selected_model == "Decision Tree":
    Hyper_parameter_tuning = st.sidebar.slider("select value for the Tuning for max depth",min_value=3,max_value=10,value=2)  
    st.write(f"Max Depth: {Hyper_parameter_tuning}")
    model = DecisionTreeRegressor(max_depth=Hyper_parameter_tuning)

elif selected_model == "Linear Regression":
    model = LinearRegression()

elif selected_model == "AdaBoost":
    Hyper_parameter_tuning = st.sidebar.slider("select value for the Tuning for max depth",min_value=3,max_value=10,value=2)
    n_estimator = st.sidebar.slider("select n_estimator for Tuning", min_value=50, max_value=150, value=50)
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)
    st.write(f"Base Max Depth: {Hyper_parameter_tuning}")
    st.write(f"Number of Estimators: {n_estimator}")
    st.write(f"Learning Rate: {learning_rate}")
    base_model = DecisionTreeRegressor(max_depth=Hyper_parameter_tuning)
    model = AdaBoostRegressor(base_model, n_estimators=n_estimator, learning_rate=learning_rate, random_state=42)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate MSE for training and testing set
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

st.subheader(f"{selected_model} Model Performance")
st.write(f"Training MSE: {mse_train:.4f}")
st.write(f"Testing MSE: {mse_test:.4f}")