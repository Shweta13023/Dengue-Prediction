#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Before running the code please ensure that the given packages are installed for the smooth running of the code:
# 1. prettytable
# 2. seaborn
# 3. tensorflow
# 4. keras
# 5. statsmodels
# 6. Prophet


# In[2]:


import warnings
import itertools
import pandas as pd
import numpy as np

from pandas.plotting import table
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error,  make_scorer

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf

from prophet import Prophet


# # Training Dataset

# ## Read Training Dataset
# 


# In[3]:


# Read data from .csv training files
dengue_features = pd.read_csv("dengue_features_train.csv")
dengue_labels = pd.read_csv("dengue_labels_train.csv")

# Merge the two datasets such that each feature row has its corresponding label
dengue = pd.merge(dengue_features, dengue_labels)
dengue.head()


# In[4]:


# Convert the week_start_date column to the DateTime data type
dengue["week_start_date"] = pd.to_datetime(dengue["week_start_date"])

# Set week_start_date column as index
dengue = dengue.set_index('week_start_date', drop=True)
dengue.head()


# In[5]:


# Plot the graph to check for different cities and their distribution in the Dataset
plt.figure(figsize = (5,5))
plt.hist(dengue["city"], bins=3, color='g')
plt.title("Distribution of cities in Training Dataset")
plt.xlabel("City")
plt.ylabel("Count")
plt.show()


# ## Split Dataset for the two Cities
# 

# In[6]:


# Split the dataset into separate cities because we don't want imputation from one city's data into another.
dengue_sj = dengue[dengue["city"] == "sj"]
dengue_iq = dengue[dengue["city"] == "iq"]

# Drop city column
dengue_sj = dengue_sj.drop("city", axis=1)
dengue_iq = dengue_iq.drop("city", axis=1)    


# In[7]:


dengue_iq.head()


# In[8]:


# Recognizing different types of features and the target variable in the datasets
time_features = ['year', 'weekofyear']
ndvi_features = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']
weather_features = ['precipitation_amt_mm', 'reanalysis_air_temp_k',
                    'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
                    'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                    'reanalysis_precip_amt_kg_per_m2',
                    'reanalysis_relative_humidity_percent', 
                    'reanalysis_sat_precip_amt_mm',
                    'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                    'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                    'station_min_temp_c', 'station_precip_mm']
target = ['total_cases']


# ## Handle Missing Values in the two Cities Datasets
# 


# In[9]:


# Function to create a Table with Columns and their Null Values for a DataFrame
def CountMissingValues(df, col_with_null, Column_label):
    NullColTable = PrettyTable(["Column", Column_label])
    for col in col_with_null:
        NullColTable.add_row([col, df[col].isnull().sum()])
    return NullColTable


# In[10]:


# Function to replace/fill all missing values using SimpleImputer
def ReplaceMissingValues(df, cols_with_null, choice_strategy):
    for col in cols_with_null:
        imputer = SimpleImputer(missing_values = np.nan, strategy = choice_strategy)
        imputer = imputer.fit(pd.DataFrame(df[col]))
        df[col] = imputer.transform(pd.DataFrame(df[col]))
    return df


# In[11]:


# Find columns with any null values
col_with_null_sj = dengue_sj.columns[dengue_sj.isnull().any()]
col_with_null_iq = dengue_iq.columns[dengue_iq.isnull().any()]


# In[12]:


# Count Null Values in each city dataset
# Call function to create tables with the count of null values in each column
NullColumnsTable_sj = CountMissingValues(dengue_sj, col_with_null_sj, "Count of Null/Missing Values for San Juan")
print(NullColumnsTable_sj)
NullColumnsTable_iq = CountMissingValues(dengue_iq, col_with_null_iq, "Count of Null/Missing Values for Iquitos")
print(NullColumnsTable_iq)


# In[13]:


# Filling these Null/Missing Values using SimpleImputer SimpleImputer
warnings.filterwarnings("ignore")

# Call function to replace/fill all missing values with median values
dengue_sj = ReplaceMissingValues(dengue_sj, col_with_null_sj, "mean")
dengue_iq = ReplaceMissingValues(dengue_iq, col_with_null_iq, "mean")


# In[14]:


# Call function to again create and print table with count of Null values in all columns and check all are zero
NullColumnsTable_sj = CountMissingValues(dengue_sj, col_with_null_sj, "Count of Null/Missing Values for San Juan")
print(NullColumnsTable_sj)
NullColumnsTable_iq = CountMissingValues(dengue_iq, col_with_null_iq, "Count of Null/Missing Values for Iquitos")
print(NullColumnsTable_iq)


# ## Handle Scales of Columns in the two cities Datasets
# 
# **Worked on By: Shweta Bhati**

# In[15]:


# Function to scale/normalize certain columns using StandardScaler
def ScaleColumns(df, cols_to_scale):
    scaler = StandardScaler()
    for col in cols_to_scale:
        scaler=scaler.fit(pd.DataFrame(df[col]))
        df[col] = scaler.transform(pd.DataFrame(df[col]))
    return df


# In[16]:


# Function to plot histograms for all scaled Weather columns in each city
def VizScaledCols(df, cols_to_scale, title):
    df[cols_to_scale].plot(kind="hist",figsize=(10,5))
    plt.title(title)
    plt.show()


# In[17]:


# Data Summary for features of San Juan
dengue_sj.describe(percentiles=[]).T


# In[18]:


# Data Summary for features of Iquitos
dengue_iq.describe(percentiles=[]).T


# In[19]:


# Call the function to Scale Weather columns for each city dataset
dengue_sj = ScaleColumns(dengue_sj, weather_features)
dengue_iq = ScaleColumns(dengue_iq, weather_features)


# In[20]:


# Data Summary for features of San Juan
dengue_sj.describe(percentiles=[]).T


# In[21]:


# Data Summary for features of Iquitos
dengue_iq.describe(percentiles=[]).T


# In[22]:


# Call function to visualize scaled weather columns in each city
VizScaledCols(dengue_sj, weather_features, "Histograms for all scaled columns in San Juan")
VizScaledCols(dengue_iq, weather_features, "Histograms for all scaled columns in Iquitos")


# # Explore and Visualize the two cities Datasets

# ## Correlation


# In[23]:


# Function to check and print correlation between features and target of a dataset
def VisualizeCorrelation(df, title):
    # Create Correlation Matrix
    corr_matrix= df.corr()

    # Visualize correlation Matrix for all columns
    plt.figure(figsize=(15, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()


# In[24]:


# Visualize the correlation between features and target using a Heatmap
VisualizeCorrelation(dengue_sj, 'Correlation Heatmap for San Juan')
VisualizeCorrelation(dengue_iq, 'Correlation Heatmap for Iquitos')


# ## Features
# 


# In[25]:


# Function to visualize the average of a type of feature values for each week in each year
def VizFeatures(df, features, mean_features, title, ylabel):
    for i in features:
        df.groupby('weekofyear')[i].mean().plot(alpha = .3, figsize = (20, 5)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for j in mean_features:
        df.groupby('weekofyear')[j].mean().plot(alpha = 1, c = 'k', linewidth = 2).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Week of Year')
    plt.show()


# ### NDVI features w.r.t 'weekofyear'

# In[26]:


dengue_sj['ndvi_north'] = dengue_sj[['ndvi_ne', 'ndvi_nw']].mean(axis=1)
dengue_sj['ndvi_south'] = dengue_sj[['ndvi_se', 'ndvi_sw']].mean(axis=1)
dengue_iq['ndvi_mean'] = dengue_iq[['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']].mean(axis=1)
ndvi_features_mean_sj = ['ndvi_north', 'ndvi_south']
ndvi_features_mean_iq = ['ndvi_mean']


# In[27]:


# Visualize the NDVI features 
VizFeatures(dengue_sj, ndvi_features, ndvi_features_mean_sj, "NDVI Levels in San Juan", "NDVI Levels")
VizFeatures(dengue_iq, ndvi_features, ndvi_features_mean_iq, "NDVI Levels in Iquitos", "NDVI Levels")


# In[28]:


dengue_sj = dengue_sj.drop("ndvi_ne", axis=1)
dengue_sj = dengue_sj.drop("ndvi_nw", axis=1)
dengue_sj = dengue_sj.drop("ndvi_se", axis=1)
dengue_sj = dengue_sj.drop("ndvi_sw", axis=1)


# In[29]:


dengue_sj.head()


# In[30]:


dengue_iq = dengue_iq.drop("ndvi_ne", axis=1)
dengue_iq = dengue_iq.drop("ndvi_nw", axis=1)
dengue_iq = dengue_iq.drop("ndvi_se", axis=1)
dengue_iq = dengue_iq.drop("ndvi_sw", axis=1)


# In[31]:


dengue_iq.head()


# ### Weather w.r.t 'weekofyear'

# In[32]:


dengue_sj['reanalysis_mean_temp_k'] = dengue_sj[['reanalysis_air_temp_k','reanalysis_avg_temp_k', 
                                     'reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k']].mean(axis=1)
dengue_iq['reanalysis_mean_temp_k'] = dengue_iq[['reanalysis_air_temp_k','reanalysis_avg_temp_k']].mean(axis=1)
weather_features_mean_sj = ['reanalysis_mean_temp_k']
weather_features_mean_iq = ['reanalysis_mean_temp_k']


# In[33]:


# Visualize the Weather features
VizFeatures(dengue_sj, weather_features, weather_features_mean_sj, "Weather Conditions in San Juan", "Values")
VizFeatures(dengue_iq, weather_features, weather_features_mean_iq, "Weather Conditions in Iquitos", "Values")


# In[34]:


dengue_sj = dengue_sj.drop("reanalysis_air_temp_k", axis=1)
dengue_sj = dengue_sj.drop("reanalysis_avg_temp_k", axis=1)
dengue_sj = dengue_sj.drop("reanalysis_dew_point_temp_k", axis=1)
dengue_sj = dengue_sj.drop("reanalysis_max_air_temp_k", axis=1)
dengue_sj = dengue_sj.drop("reanalysis_min_air_temp_k", axis=1)


# In[35]:


dengue_sj.head()


# In[36]:


dengue_iq = dengue_iq.drop("reanalysis_air_temp_k", axis=1)
dengue_iq = dengue_iq.drop("reanalysis_avg_temp_k", axis=1)


# In[37]:


dengue_iq.head()


# ## Target


# In[38]:


# Features now
time_features = ['year', 'weekofyear']
sj_features = ['precipitation_amt_mm', 'reanalysis_precip_amt_kg_per_m2',
            'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
            'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
            'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
            'station_min_temp_c', 'station_precip_mm', 
            'total_cases', 
            'ndvi_north', 'ndvi_south', 
            'reanalysis_mean_temp_k']
iq_features = ['reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
               'precipitation_amt_mm', 'reanalysis_precip_amt_kg_per_m2',
               'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
               'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
               'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
               'station_min_temp_c', 'station_precip_mm', 
               'total_cases', 
               'ndvi_mean', 
               'reanalysis_mean_temp_k']
ndvi_sj_features = ['ndvi_north', 'ndvi_south']
ndvi_iq_features = ['ndvi_mean']
weather_sj_features = ['precipitation_amt_mm', 'reanalysis_precip_amt_kg_per_m2',
                    'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
                    'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                    'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                    'station_min_temp_c', 'station_precip_mm','reanalysis_mean_temp_k']
weather_iq_features = ['precipitation_amt_mm', 'reanalysis_precip_amt_kg_per_m2',
                    'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                    'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
                    'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                    'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                    'station_min_temp_c', 'station_precip_mm','reanalysis_mean_temp_k']
target = ['total_cases']


# ### Target w.r.t NDVI and Weather Features

# In[39]:


# Function to check the correlation of target features with NDVI and Weather features
def VisualizeTargetCorrelation(df,title):
    # Create Correlation Matrix
    corr_matrix= df.corr(numeric_only=True)

    # Show Correlation for Total Cases
    plt.figure(figsize=(10, 4))
    corr_matrix[target].drop(target).sort_values(by = target, ascending=False).plot.barh()
    plt.title(title)
    plt.show()


# In[40]:


# Visualize the correlation of target with NDVI and Weather features
VisualizeTargetCorrelation(dengue_sj[sj_features], "Correlation of all features with target for San Juan")
VisualizeTargetCorrelation(dengue_iq[iq_features], "Correlation of all features with target for Iquitos")


# ### Target w.r.t 'year' and 'weekofyear'

# In[41]:


# Function to visualize the total cases w.r.t 'year' and 'weekofyear'
def VizTarget1(df,title):
    df2 = df
    df2 = df2.reset_index()
    df2 = df2.set_index(['year','weekofyear'], drop=True)
    df2[target].plot(figsize = (10,4))
    plt.title(title)
    plt.xlabel("Year, Week of Year")
    plt.ylabel("Number of Cases")
    plt.show()


# In[42]:


# Visualize the total cases w.r.t 'year' and 'weekofyear'
VizTarget1(dengue_sj,"Number of Cases in San Juan")
VizTarget1(dengue_iq,"Number of Cases in Iquitos")


# In[43]:


# Function to visualize the decomposed target values
def VizDecomposedTarget(df):
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(df[target], model='additive', period=52)

    # Plot the decomposition
    plt.figure(figsize=(15, 10))
    decomposition.plot()
    plt.show()


# In[44]:


# Visualize the decomposed target values
print("For San Juan:")
VizDecomposedTarget(dengue_sj)
print("For Iquitos:")
VizDecomposedTarget(dengue_iq)


# ### Target w.r.t 'weekofyear' for all years

# In[45]:


# Function to visualize the total cases w.r.t 'weekofyear' for all years
def VizTarget2(df, title):
    df2 = df
    df2 = df2.reset_index()
    df2 = df2.set_index(['year','weekofyear'], drop=True)
    # Find all unique values in the first level of the index
    unique_years = df2.index.get_level_values(0).unique()
    for i in unique_years:
        df3 = df2.loc[i]
        plt.plot(df3[target], alpha = .3)

    df2.groupby('weekofyear')['total_cases'].mean().plot(c = 'k', figsize = (10,4))
    plt.legend(unique_years, loc='center left', bbox_to_anchor=(1, .5))
    plt.title(title)
    plt.xlabel('Week of the Year')
    plt.ylabel('Number of Cases')
    plt.show()


# In[46]:


# Visualize the total cases w.r.t 'weekofyear' for all years
VizTarget2(dengue_sj, "Number of Cases per Week in San Juan")
VizTarget2(dengue_iq, "Number of Cases per Week in Iquitos")


# # Training Data Modelling and Testing

# ## Split Training Dataset into Train Data for Modeling and Test Data for Testing
# 


# In[47]:


# Split San Juan Dataset
dengue_sj.sort_index()

X_sj = dengue_sj.drop(columns=target, axis=1)
y_sj = dengue_sj[target]

sj_train_start_index = dengue_sj.index[0]
sj_train_end_index = dengue_sj.index[(int(len(dengue_sj)*0.7))+3]
sj_test_start_index = dengue_sj.index[(int(len(dengue_sj)*0.7))+4]
sj_test_end_index = dengue_sj.index[-1]

dengue_sj_train = dengue_sj.loc[sj_train_start_index:sj_train_end_index]
X_sj_train = dengue_sj_train.drop(columns=target, axis=1)
y_sj_train = dengue_sj_train[target]

dengue_sj_test = dengue_sj.loc[sj_test_start_index:sj_test_end_index]
X_sj_test = dengue_sj_test.drop(columns=target, axis=1)
y_sj_test = dengue_sj_test[target]


# In[48]:


# Split Iquitos Dataset
dengue_iq.sort_index()

X_iq = dengue_iq.drop(columns=target, axis=1)
y_iq = dengue_iq[target]

iq_train_start_index = dengue_iq.index[0]
iq_train_end_index = dengue_iq.index[(int(len(dengue_iq)*0.7))+4]
iq_test_start_index = dengue_iq.index[(int(len(dengue_iq)*0.7))+5]
iq_test_end_index = dengue_iq.index[-1]

dengue_iq_train = dengue_iq.loc[iq_train_start_index:iq_train_end_index]
X_iq_train = dengue_iq_train.drop(columns=target, axis=1)
y_iq_train = dengue_iq_train[target]

dengue_iq_test = dengue_iq.loc[iq_test_start_index:iq_test_end_index]
X_iq_test = dengue_iq_test.drop(columns=target, axis=1)
y_iq_test = dengue_iq_test[target]


# ## Regression Modeling and Testing
# 



# In[49]:


def FitModelandEvaluate(model, param_grid, X_train, y_train):
    # Define k-folds for Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define mean squared error as the scoring metric
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    
    # Perform GridSearchCV with K-fold cross-validation for hyperparameter tuning
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, error_score='raise', cv=kf)
    grid_result = grid.fit(X_train, y_train)

    # Get the best model
    best_model = grid_result.best_estimator_

    # Evaluate the best model using cross-validation
    cv_results = grid_result.cv_results_
    print("Cross-validation results:")
    for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
        print(f"Mean MSE: {-mean_score:.4f}, Params: {params}")

    # You can also extract the best parameters and the corresponding MSE
    best_params = grid_result.best_params_
    best_mse = -grid_result.best_score_
    print("\nBest Parameters:", best_params)
    print("Best Mean MSE:", best_mse)
    return best_mse, best_params, best_model


# In[50]:


# Initalize different Regression Models and their parameter grids

# Initialize the linear regression model
model_1 = LinearRegression()
# Define hyperparameters grid for tuning
param_grid_1 = {}

# Initialize the Random Forest regression model
model_2 = RandomForestRegressor(random_state=42)
# Define hyperparameters grid for tuning
param_grid_2 = {
    'n_estimators': [100, 200, 300]
}

# Initialize the Gradient Boosting regression model
model_3 = GradientBoostingRegressor(random_state=42)
# Define hyperparameters grid for tuning
param_grid_3 = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Initialize the SVR model
model_4 = SVR()
# Define hyperparameters grid for tuning
param_grid_4 = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10]
}

# Define the neural network model for San Juan
def create_model_sj(neurons=1, activation='tanh', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(X_sj.shape[1],)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1)) 
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def create_model_iq(neurons=1, activation='tanh', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(X_iq.shape[1],)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1)) 
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
    
# Create KerasRegressor wrapper for scikit-learn
model_sj_5 = KerasRegressor(build_fn=create_model_sj, verbose=0)
model_iq_5 = KerasRegressor(build_fn=create_model_iq, verbose=0)
# Define hyperparameters grid for tuning
param_grid_5 = {
    'neurons': [10, 50, 100],
    'optimizer': ['adam', 'sgd']
}


# In[51]:


#Visualise all MSE values on the graph for comparison to choose the best model
def VizMSEValues(mse_1, mse_2, mse_3, mse_4, mse_5, param_1, param_2, param_3, param_4, param_5, model_1, model_2, model_3, model_4, model_5):
    mse_values = PrettyTable(["Model", "MSE", "Parameters"])
    mse_values.add_row(['Linear Regression', mse_1, param_1])
    mse_values.add_row(['Random Forest Regression', mse_2, param_2])
    mse_values.add_row(['Gradient Boost Regression', mse_3, param_3])
    mse_values.add_row(['Support Vector Regression', mse_4, param_4])
    mse_values.add_row(['Neural Network Regression', mse_5, param_5])
    
    print(mse_values)

    mse_table= pd.DataFrame([['Linear Regression', mse_1, model_1],
                              ['RandomForest Regression', mse_2, model_2], 
                              ['Gradient Boost Regression',mse_3, model_3],
                              ['Support Vector Regression', mse_4, model_4], 
                              ['Neural Network Regression', mse_5, model_5]])
    
    
    plt.figure(figsize=(12,3))
    plt.scatter(mse_table[0], mse_table[1],  color='blue')
    plt.plot(mse_table[0], mse_table[1], color='red', linestyle='-', linewidth=1)
    plt.xlabel('Models')
    plt.ylabel('MSE')
    plt.title('MSE Values for all Regression Models')
    plt.show()

    best_model = mse_table[mse_table[1]==min(mse_table[1])]
    return best_model


# In[52]:


# Plotting the actual vs predicted values
def VizPredictedvsActual(y_pred, y_test, title):
    plt.figure(figsize=(15, 5))
    y_pred['predicted_cases'].plot(label='Predictions', color='blue', marker='o')
    y_test['total_cases'].plot(label='Actual Values', color='red', marker='x')
    plt.title(title)
    
    plt.xlabel('Year')
    plt.ylabel('Total Cases')
    plt.legend()
    plt.grid(True)
    plt.show()


# ### San Juan

# In[53]:


print("Linear Regression:")
mse_sj_1, params_sj_1, model_sj_1 = FitModelandEvaluate(model_1, param_grid_1, X_sj_train, y_sj_train)


# In[54]:


print("Random Forest Regression:")
mse_sj_2, params_sj_2, model_sj_2 = FitModelandEvaluate(model_2, param_grid_2, X_sj_train, y_sj_train)


# In[55]:


print("Gradient Boost Regression:")
mse_sj_3, params_sj_3, model_sj_3 = FitModelandEvaluate(model_3, param_grid_3, X_sj_train, y_sj_train)


# In[56]:


print("Support Vector Regression:")
mse_sj_4, params_sj_4, model_sj_4 = FitModelandEvaluate(model_4, param_grid_4, X_sj_train, y_sj_train)


# In[57]:


print("Neural Netwrok Regression:")
mse_sj_5, params_sj_5, model_sj_5 = FitModelandEvaluate(model_sj_5, param_grid_5, X_sj_train, y_sj_train)


# In[58]:


# Visualize the mse values
best_model_sj = VizMSEValues(mse_sj_1, mse_sj_2, mse_sj_3, mse_sj_4, mse_sj_5, 
             params_sj_1, params_sj_2, params_sj_3, params_sj_4, params_sj_5, 
             model_sj_1, model_sj_2, model_sj_3, model_sj_4, model_sj_5)


# In[59]:


# Best Chosen Model
regression_model = best_model_sj[2][2]
regression_model


# In[60]:


# Make predictions with best chosen model
y_sj_predictions = pd.Series(regression_model.predict(X_sj_test))


# In[61]:


y_sj_predictions = y_sj_predictions.astype(int)
y_sj_predictions.index = y_sj_test.index
y_sj_predictions = y_sj_predictions.to_frame(name='predicted_cases')


# In[62]:


# Vizualize the Actual vs Predicted Number of Cases on the Test Split of San Juan
VizPredictedvsActual (y_sj_predictions, y_sj_test, "Predictions vs Actual Target Values of Test Split for San Juan using Regression")


# In[63]:


# Evaluate the Best Model
mse_sj = mean_squared_error(y_sj_predictions, y_sj_test)
print("MSE for the Gradient Boost Regression model for San Juan Dataset: ", mse_sj)


# ### Iquitos

# In[64]:


print("Linear Regression:")
mse_iq_1, params_iq_1, model_iq_1 = FitModelandEvaluate(model_1, param_grid_1, X_iq_train, y_iq_train)


# In[65]:


print("Random Forest Regression:")
mse_iq_2, params_iq_2, model_iq_2 = FitModelandEvaluate(model_2, param_grid_2, X_iq_train, y_iq_train)


# In[66]:


print("Gradient Boost Regression:")
mse_iq_3, params_iq_3, model_iq_3 = FitModelandEvaluate(model_3, param_grid_3, X_iq_train, y_iq_train)


# In[67]:


print("Support Vector Regression:")
mse_iq_4, params_iq_4, model_iq_4 = FitModelandEvaluate(model_4, param_grid_4, X_iq_train, y_iq_train)


# In[68]:


print("Neural Network Regression:")
mse_iq_5, params_iq_5, model_iq_5 = FitModelandEvaluate(model_iq_5, param_grid_5, X_iq_train, y_iq_train)


# In[69]:


# Visualize the mse values
best_model_iq = VizMSEValues(mse_iq_1, mse_iq_2, mse_iq_3, mse_iq_4, mse_iq_5, 
                             params_iq_1, params_iq_2, params_iq_3, params_iq_4, params_iq_5,
                             model_iq_1, model_iq_2, model_iq_3, model_iq_4, model_iq_5)


# In[70]:


# Best chosen model
regression_model_2 = best_model_iq[2][1]


# In[71]:


# Make predictions with the best chosen model
y_iq_predictions = pd.Series(regression_model_2.predict(X_iq_test))


# In[72]:


y_iq_predictions = y_iq_predictions.astype(int)
y_iq_predictions.index = y_iq_test.index
y_iq_predictions = y_iq_predictions.to_frame(name='predicted_cases')


# In[73]:


# Vizualize the Actual vs Predicted Number of Cases on the Test Split of Iquitos
VizPredictedvsActual(y_iq_predictions, y_iq_test, "Predictions vs Actual Target Values of Test Split for Iquitos using Regression")


# In[74]:


# Evaluate the Best Model
mse_iq = mean_squared_error(y_iq_predictions, y_iq_test)
print("MSE for the Random forest Regression model for Iquitos Dataset: ", mse_iq)


# ## Time Series Modelling


# In[75]:


#function to get the best parameters for ARIMA model
def BestParamARIMA(y_train):
    # Define the range of values for p, d, and q
    p_values = range(0, 3) 
    d_values = range(0, 3) 
    q_values = range(0, 3) 

    # Generate all possible combinations of p, d, and q
    pdq_values = list(itertools.product(p_values, d_values, q_values))

    # Evaluate each combination using AIC or BIC and select the best one
    best_aic = float("inf")
    best_params = None

    for pdq in pdq_values:
        try:
            model = ARIMA(y_train, order=pdq)
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic = aic
                best_params = pdq
        except:
            continue
    print("Best ARIMA parameters:", best_params)
    print("Best AIC:", best_aic)
    return best_params


# In[76]:


#function to visualize actual and predicted values
def VizPredictedvsActualARIMA(y_pred, y_test, title):
    plt.figure(figsize=(15, 5))
    y_pred['predicted_mean'].plot(label='Predictions', color='blue', marker='X')
    y_test['total_cases'].plot(label='Actual Values', color='red', marker='x')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Total Cases')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[77]:


#function to get the best parameters for Negative Binomial model
def GetStatsNegBinomialModel(train, test, model_formula):
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model


# In[78]:


#function to visualize actual and predicted values
def VizPredictedvsActualStatsNegBin(df, best_model, title):
    figs, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    df['fitted'] = best_model.fittedvalues
    df.fitted.plot(ax=axes, label="Predictions")
    df.total_cases.plot(ax=axes, label="Actual")
    plt.xlabel("Year")
    plt.ylabel("Total Cases")
    plt.title(title)
    plt.show()


# In[79]:


#function to get the best parameters for Prophet model
def GetProphet(df):
    df2 = df
    df2 = df2.reset_index()
    df2.rename(columns={'week_start_date': 'ds','total_cases': 'y'}, inplace=True)

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df2)
    return model


# In[80]:


#function to visualize actual and predicted values 
def VizPredictedvsActualProphet(y_pred, y_test, title):
    plt.figure(figsize=(15, 5))
    y_pred['yhat'].plot(label='Predictions', color='blue', marker='X')
    y_test['total_cases'].plot(label='Actual Values', color='red', marker='x')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Total Cases')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[113]:


#visualise to compare MSE values and choose the best model
def VizMSEValues2(mse_1, mse_2, mse_3, model_1, model_2, model_3):
    mse_values = PrettyTable(["Model", "MSE"])
    mse_values.add_row(['ARIMA', mse_1])
    mse_values.add_row(['Stats Negative Binomial', mse_2])
    mse_values.add_row(['Prophet', mse_3])
    
    print(mse_values)

    mse_table= pd.DataFrame([['ARIMA', mse_1, model_1],
                             ['Stats Negative Binomial', mse_2, model_2],
                             ['Prophet', mse_3, model_3]])
    
    
    plt.figure(figsize=(12,3))
    plt.scatter(mse_table[0], mse_table[1],  color='blue')
    plt.plot(mse_table[0], mse_table[1], color='red', linestyle='-', linewidth=1)
    plt.xlabel('Models')
    plt.ylabel('MSE')
    plt.title('MSE Values for all Statistical Models')
    plt.show()

    best_model = mse_table[mse_table[1]==min(mse_table[1])]
    return best_model


# ### San Juan

# In[82]:


# ARIMA

# Find the best parameters for ARIMA 
best_param_sj = BestParamARIMA(y_sj_train)


# In[83]:


# Fit the ARIMA model with best parameters
arima_model_sj = ARIMA(y_sj_train, order=best_param_sj)
arima_model_sj_fit = arima_model_sj.fit()
print(arima_model_sj_fit.summary())


# In[84]:


# Making Predictions
y_sj_pred = arima_model_sj_fit.forecast(steps=len(y_sj_test))
y_sj_pred.index = y_sj_test.index
y_sj_pred = pd.DataFrame(y_sj_pred)
y_sj_pred['predicted_mean'] = y_sj_pred['predicted_mean'].astype(int)


# In[85]:


# Visualize Predictions
VizPredictedvsActualARIMA(y_sj_pred, y_sj_test, "Predictions vs Actual Target Values of Test Split for San Juan using ARIMA")


# In[86]:


# Evaluate the Predictions
mse_sj_t1 = mean_squared_error(y_sj_pred['predicted_mean'], y_sj_test['total_cases'])
print("MSE for the ARIMA TIme Series model for San Juan Dataset: ", mse_sj_t1)


# In[87]:


# Stats Negative Binomial

# Specify the formule of the model for Stats Negative Binomial
model_formula_sj = "total_cases ~ 1 +"                     "reanalysis_specific_humidity_g_per_kg +"                     "station_avg_temp_c +"                     "reanalysis_mean_temp_k"

# Fit Stats Negative Binomial Model for best parameters
stats_model_sj = GetStatsNegBinomialModel(dengue_sj_train, dengue_sj_test,model_formula_sj)


# In[88]:


# Best Stats Model summary
print(stats_model_sj.summary())


# In[89]:


# Make Predictions and Visualize
VizPredictedvsActualStatsNegBin(dengue_sj_test, stats_model_sj, "Predictions vs Actual Target Values of Test Split for San Juan using Stats Negative Binomial")


# In[90]:


# Evaluate the Predicitons
mse_sj_t2 = mean_squared_error(dengue_sj_test['fitted'].astype(int), dengue_sj_test['total_cases'])
print("MSE for the Stats Negative Binomial Time Series model for San Juan Dataset: ", mse_sj_t2)


# In[91]:


# Prophet

# Get the prophet model for improvised train dataset
prophet_model_sj = GetProphet(y_sj_train)


# In[92]:


# Get Test data ready for Prophet Model fitting
y_sj_test_rev = y_sj_test
y_sj_test_rev = y_sj_test_rev.reset_index()
y_sj_test_rev.rename(columns={'week_start_date': 'ds'}, inplace=True)

# Make predictions
y_sj_prophet_pred = prophet_model_sj.predict(y_sj_test_rev)


# In[93]:


# Get Test data ready
y_sj_prophet_pred.index = y_sj_test.index
y_sj_prophet_pred['yhat'] = y_sj_prophet_pred['yhat'].astype(int)


# In[94]:


# Visualize Predictions
VizPredictedvsActualProphet(y_sj_prophet_pred, y_sj_test, "Predictions vs Actual Target Values of Test Split for San Juan using Prophet")


# In[95]:


# Evaluate the Predictions
mse_sj_t3 = mean_squared_error(y_sj_prophet_pred['yhat'], dengue_sj_test['total_cases'])
print("MSE for the Stats Negative Binomial Time Series model for San Juan Dataset: ", mse_sj_t3)


# In[114]:


# Visualize the mse to find best model
best_model_sj2 = VizMSEValues2(mse_sj_t1, mse_sj_t2, mse_sj_t3, arima_model_sj, stats_model_sj, prophet_model_sj)


# In[97]:


best_ts_model_sj = best_model_sj2[2][1]
best_ts_model_sj


# ### Iquitos

# In[98]:


# ARIMA

# Find the best parameters for ARIMA 
best_param_iq = BestParamARIMA(y_iq_train)


# In[99]:


# Fit the ARIMA model with best parameters
arima_model_iq = ARIMA(y_iq_train, order=best_param_sj)
arima_model_iq_fit = arima_model_iq.fit()
print(arima_model_iq_fit.summary())


# In[100]:


# Making Predictions with best ARIMA Model with best parameters
# Validate the model
y_iq_pred = arima_model_iq_fit.forecast(steps=len(y_iq_test))
y_iq_pred.index = y_iq_test.index
y_iq_pred = pd.DataFrame(y_iq_pred)
y_iq_pred['predicted_mean'] = y_iq_pred['predicted_mean'].astype(int)


# In[101]:


VizPredictedvsActualARIMA(y_iq_pred, y_iq_test, "Predictions vs Actual Target Values of Test Split for Iquitos using ARIMA")


# In[102]:


# Evaluate the Best Model
mse_iq_t1 = mean_squared_error(y_iq_pred['predicted_mean'], y_iq_test['total_cases'])
print("MSE for the ARIMA TIme Series model for Iquitos Dataset: ", mse_iq_t1)


# In[103]:


# Stats Negative Binomial

# Find Best Parameters for Stats Negative Binomial Model
# Specify the formule of the model for Stats Negative Binomial
model_formula_iq = "total_cases ~ 1 +"                     "reanalysis_specific_humidity_g_per_kg +"                     "reanalysis_dew_point_temp_k +"                     "reanalysis_min_air_temp_k"
stats_model_iq = GetStatsNegBinomialModel(dengue_iq_train, dengue_iq_test, model_formula_iq)


# In[104]:


# Find the best Stats Model summary
print(stats_model_iq.summary())


# In[105]:


VizPredictedvsActualStatsNegBin(dengue_iq_test, stats_model_iq, "Predictions vs Actual Target Values of Test Split for Iquitos using Stats Negative Binomial")


# In[106]:


# Evaluate the Best Model
mse_iq_t2 = mean_squared_error(dengue_iq_test['fitted'].astype(int), dengue_iq_test['total_cases'])
print("MSE for the Stats Negative Binomial Time Series model for San Juan Dataset: ", mse_iq_t2)


# In[107]:


# Prophet

# Get the prophet model for improvised train dataset
prophet_model_iq = GetProphet(y_iq_train)


# In[108]:


# Get Test data ready for Prophet Model fitting
y_iq_test_rev = y_iq_test
y_iq_test_rev = y_iq_test_rev.reset_index()
y_iq_test_rev.rename(columns={'week_start_date': 'ds'}, inplace=True)

# Make predictions
y_iq_prophet_pred = prophet_model_iq.predict(y_iq_test_rev)


# In[109]:


# Get Test data ready
y_iq_prophet_pred.index = y_iq_test.index
y_iq_prophet_pred['yhat'] = y_iq_prophet_pred['yhat'].astype(int)


# In[110]:


# Visualize Predictions
VizPredictedvsActualProphet(y_iq_prophet_pred, y_iq_test, "Predictions vs Actual Target Values of Test Split for Iquitos using Prophet")


# In[111]:


# Evaluate the Predictions
mse_iq_t3 = mean_squared_error(y_iq_prophet_pred['yhat'], dengue_iq_test['total_cases'])
print("MSE for the Stats Negative Binomial Time Series model for San Juan Dataset: ", mse_iq_t3)


# In[116]:


# Visualize the MSE to find the best model
best_model_iq2 = VizMSEValues2(mse_iq_t1, mse_iq_t2, mse_iq_t3, arima_model_iq, stats_model_iq, prophet_model_iq)


# In[117]:


best_ts_model_iq = best_model_iq2[2][1]
best_ts_model_iq


# # Final Testing Dataset

# ## Read Testing Dataset


# In[118]:


# Preprocess the test data
# Read data from .csv test files
dengue_final_test = pd.read_csv("dengue_features_test.csv")

# Convert the date column to the DateTime data type
dengue_final_test["week_start_date"] = pd.to_datetime(dengue_final_test["week_start_date"])

# Set index to the year and weekofyear column
dengue_final_test = dengue_final_test.set_index('week_start_date', drop=True)
dengue_final_test.head()


# ## Split Dataset for the two Cities
# 


# In[119]:


# Split data into separate cities because we don't want imputation from one city's data into another.
dengue_final_test_sj = dengue_final_test[dengue_final_test["city"] == "sj"]
dengue_final_test_iq = dengue_final_test[dengue_final_test["city"] == "iq"]
# Drop city column
dengue_final_test_sj = dengue_final_test_sj.drop("city", axis=1)
dengue_final_test_iq = dengue_final_test_iq.drop("city", axis=1)


# In[120]:


dengue_final_test_sj.head()


# In[121]:


dengue_final_test_iq.head()


# ## Handle Missing Values in the two Cities Datasets
# 


# In[122]:


# Cleaning
# Find columns with any null values to be passed to function that counts the number of null values and prints it in a table
col_with_null_sj = dengue_final_test_sj.columns[dengue_final_test_sj.isnull().any()]
col_with_null_iq = dengue_final_test_iq.columns[dengue_final_test_iq.isnull().any()]

# Call function to again create and print table with count of Null values in all columns and check all are zero
NullColumnsTable_sj = CountMissingValues(dengue_final_test_sj, col_with_null_sj, "Count of Null/Missing Values for San Juan")
print(NullColumnsTable_sj)
NullColumnsTable_iq = CountMissingValues(dengue_final_test_iq, col_with_null_iq, "Count of Null/Missing Values for Iquitos")
print(NullColumnsTable_iq)


# In[123]:


# Call function to replace/fill all missing values with median values
dengue_final_test_sj = ReplaceMissingValues(dengue_final_test_sj, col_with_null_sj, "mean")
dengue_final_test_iq = ReplaceMissingValues(dengue_final_test_iq, col_with_null_iq, "mean")


# In[124]:


# Call function to again create and print table with count of Null values in all columns and check all are zero
NullColumnsTable_sj = CountMissingValues(dengue_final_test_sj, col_with_null_sj, "Count of Null/Missing Values for San Juan")
print(NullColumnsTable_sj)
NullColumnsTable_iq = CountMissingValues(dengue_final_test_iq, col_with_null_iq, "Count of Null/Missing Values for Iquitos")
print(NullColumnsTable_iq)


# ## Handle Scales of Columns in the two cities Datasets
# 
# **Worked on By: Shweta Bhati**

# In[125]:


# Scaling
# Call the function to scale selected columns for each city dataset
dengue_final_test_sj = ScaleColumns(dengue_final_test_sj, weather_features)
dengue_final_test_iq = ScaleColumns(dengue_final_test_iq, weather_features)


# In[126]:


# Call function to plot histograms for scaled columns in each city
VizScaledCols(dengue_final_test_sj, weather_features, "Histograms for all scaled columns in San Juan")
VizScaledCols(dengue_final_test_iq, weather_features, "Histograms for all scaled columns in Iquitos")


# ## Feature Extraction
# 


# In[127]:


dengue_final_test_sj['ndvi_north'] = dengue_final_test_sj[['ndvi_ne', 'ndvi_nw']].mean(axis=1)
dengue_final_test_sj['ndvi_south'] = dengue_final_test_sj[['ndvi_se', 'ndvi_sw']].mean(axis=1)
dengue_final_test_iq['ndvi_mean'] = dengue_final_test_iq[['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']].mean(axis=1)

dengue_final_test_sj['reanalysis_mean_temp_k'] = dengue_final_test_sj[['reanalysis_air_temp_k','reanalysis_avg_temp_k', 
                                     'reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k']].mean(axis=1)
dengue_final_test_iq['reanalysis_mean_temp_k'] = dengue_final_test_iq[['reanalysis_air_temp_k','reanalysis_avg_temp_k']].mean(axis=1)


# In[128]:


dengue_final_test_sj = dengue_final_test_sj.drop("ndvi_ne", axis=1)
dengue_final_test_sj = dengue_final_test_sj.drop("ndvi_nw", axis=1)
dengue_final_test_sj = dengue_final_test_sj.drop("ndvi_se", axis=1)
dengue_final_test_sj = dengue_final_test_sj.drop("ndvi_sw", axis=1)

dengue_final_test_iq = dengue_final_test_iq.drop("ndvi_ne", axis=1)
dengue_final_test_iq = dengue_final_test_iq.drop("ndvi_nw", axis=1)
dengue_final_test_iq = dengue_final_test_iq.drop("ndvi_se", axis=1)
dengue_final_test_iq = dengue_final_test_iq.drop("ndvi_sw", axis=1)

dengue_final_test_sj = dengue_final_test_sj.drop("reanalysis_air_temp_k", axis=1)
dengue_final_test_sj = dengue_final_test_sj.drop("reanalysis_avg_temp_k", axis=1)
dengue_final_test_sj = dengue_final_test_sj.drop("reanalysis_dew_point_temp_k", axis=1)
dengue_final_test_sj = dengue_final_test_sj.drop("reanalysis_max_air_temp_k", axis=1)
dengue_final_test_sj = dengue_final_test_sj.drop("reanalysis_min_air_temp_k", axis=1)

dengue_final_test_iq = dengue_final_test_iq.drop("reanalysis_air_temp_k", axis=1)
dengue_final_test_iq = dengue_final_test_iq.drop("reanalysis_avg_temp_k", axis=1)


# In[129]:


dengue_final_test_sj.head()


# In[130]:


dengue_final_test_iq.head()


# # Regression Predictions on Final Datatset
# 


# In[131]:


# Make predictions
dengue_final_test_sj_preds = pd.Series(regression_model.predict(dengue_final_test_sj))
dengue_final_test_iq_preds = pd.Series(regression_model_2.predict(dengue_final_test_iq))


# In[132]:


dengue_final_test_sj_preds = dengue_final_test_sj_preds.astype(int)
dengue_final_test_sj_preds.index = dengue_final_test_sj.index
dengue_final_test_iq_preds = dengue_final_test_iq_preds.astype(int)
dengue_final_test_iq_preds.index = dengue_final_test_iq.index


# In[133]:


def VizFinalPred(pred, title):
    plt.figure(figsize=(10, 5))
    pred.plot(color='green', marker='o')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Total Cases')
    plt.grid(True)
    plt.show()


# In[134]:


VizFinalPred(dengue_final_test_sj_preds, "San Juan Predictions for Total Cases on Final Test Dataset using Regression")
VizFinalPred(dengue_final_test_iq_preds, "Iquitos Predictions for Total Cases on Final Test Dataset using Regression")


# In[135]:


# add column name to predictions
dengue_test_sj_preds_final = dengue_final_test_sj_preds.to_frame(name='total_cases')
dengue_test_iq_preds_final = dengue_final_test_iq_preds.to_frame(name='total_cases')
dengue_test_sj_preds_final['city'] = 'sj'
dengue_test_iq_preds_final['city'] = 'iq'
dengue_test_sj_preds_final = pd.merge(dengue_final_test_sj[dengue_final_test_sj.columns[0:2]], dengue_test_sj_preds_final, left_index=True, right_index=True)
dengue_test_iq_preds_final = pd.merge(dengue_final_test_iq[dengue_final_test_sj.columns[0:2]], dengue_test_iq_preds_final, left_index=True, right_index=True)
dengue_test_sj_preds_final = dengue_test_sj_preds_final[['city', 'year', 'weekofyear', 'total_cases']]
dengue_test_iq_preds_final = dengue_test_iq_preds_final[['city', 'year', 'weekofyear', 'total_cases']]


# In[136]:


dengue_test_preds_final = pd.concat([dengue_test_sj_preds_final, dengue_test_iq_preds_final])


# In[137]:


dengue_test_preds_final.to_csv('Test_Data_Regression_Preditions.csv', index=False)


# # Time Series Predictions on Final Dataset
# 
# **Worked on By: Shweta Bhati**

# In[138]:


# Make predictions
dengue_test_sj_stats_preds = pd.Series(best_ts_model_sj.predict(dengue_final_test_sj).astype(int))
dengue_test_iq_stats_preds = pd.Series(best_ts_model_iq.predict(dengue_final_test_iq).astype(int))


# In[139]:


VizFinalPred(dengue_test_sj_stats_preds, "San Juan Predictions for Total Cases on Final Test Dataset using Stats Negative Binomial")
VizFinalPred(dengue_test_iq_stats_preds, "Iquitos Predictions for Total Cases on Final Test Dataset using Stats Negative Binomial")


# In[140]:


# add column name to predictions
dengue_test_sj_stats_preds_final = dengue_test_sj_stats_preds.to_frame(name='total_cases')
dengue_test_iq_stats_preds_final = dengue_test_iq_stats_preds.to_frame(name='total_cases')
dengue_test_sj_stats_preds_final['city'] = 'sj'
dengue_test_iq_stats_preds_final['city'] = 'iq'
dengue_test_sj_stats_preds_final = pd.merge(dengue_final_test_sj[dengue_final_test_sj.columns[0:2]], dengue_test_sj_stats_preds_final, left_index=True, right_index=True)
dengue_test_iq_stats_preds_final = pd.merge(dengue_final_test_iq[dengue_final_test_sj.columns[0:2]], dengue_test_iq_stats_preds_final, left_index=True, right_index=True)
dengue_test_sj_stats_preds_final = dengue_test_sj_preds_final[['city', 'year', 'weekofyear', 'total_cases']]
dengue_test_iq_stats_preds_final = dengue_test_iq_preds_final[['city', 'year', 'weekofyear', 'total_cases']]


# In[141]:


dengue_test_stats_preds_final = pd.concat([dengue_test_sj_stats_preds_final, dengue_test_iq_stats_preds_final])


# In[142]:


dengue_test_stats_preds_final.to_csv('Test_Data_Time_Series_Preditions.csv', index=False)

