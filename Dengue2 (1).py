#!/usr/bin/env python
# coding: utf-8

# In[130]:


import warnings

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


# # Read the dataset

# In[91]:


# Read data from .csv training files
dengue_features = pd.read_csv("dengue_features_train.csv")
dengue_labels = pd.read_csv("dengue_labels_train.csv")
# Merge the two training datasets such it each feature's row has it's corresponding label
dengue = pd.merge(dengue_features, dengue_labels)
# Convert the date column to the DateTime data type
dengue["week_start_date"] = pd.to_datetime(dengue["week_start_date"])
# Drop duplicate rows from the training dataset
dengue = dengue.drop_duplicates()


# In[95]:


plt.figure(figsize = (5,5))
plt.hist(dengue["city"], bins=3, color='g')
plt.title("Distribution of cities in Training Dataset")
plt.xlabel("City")
plt.ylabel("Count")
plt.show()


# In[172]:


# Split data into seperate cities because we don't want imputation from one city's data into another.
dengue_sj = dengue[dengue["city"] == "sj"]
dengue_iq = dengue[dengue["city"] == "iq"]


# In[173]:


# set index to the dates
dengue_sj.set_index('week_start_date', drop = True, inplace = True)
dengue_iq.set_index('week_start_date', drop = True, inplace = True)


# # Clean the dataset

# ## Find and Fill Null Values 

# In[174]:


# Count Null Values in each city dataset

# Function to create a Table with Columns and their Null Values for a DataFrame
def CountMissingValues(df, col_with_null, Column_label):
    NullColTable = PrettyTable(["Column", Column_label])
    for col in col_with_null:
        NullColTable.add_row([col, df[col].isnull().sum()])
    return NullColTable

# Find columns with any null values to be passed to function that counts the number of null values and prints it in a table
col_with_null_sj = dengue_sj.columns[dengue_sj.isnull().any()]
col_with_null_iq = dengue_iq.columns[dengue_iq.isnull().any()]

# Call function to create tables with the count of null values in each column
NullColumnsTable_sj = CountMissingValues(dengue_sj, col_with_null_sj, "Count of Null/Missing Values for San Juan")
print(NullColumnsTable_sj)
NullColumnsTable_iq = CountMissingValues(dengue_iq, col_with_null_iq, "Count of Null/Missing Values for Iquitos")
print(NullColumnsTable_iq)


# In[175]:


# Filling these Null/Missing Values using SimpleImputer SimpleImputer
warnings.filterwarnings("ignore")

# Function to replace/fill all missing values using SimpleImputer
def ReplaceMissingValues(df, cols_with_null, choice_strategy):
    for col in cols_with_null:
        imputer = SimpleImputer(missing_values = np.nan, strategy = choice_strategy)
        imputer = imputer.fit(pd.DataFrame(df[col]))
        df[col] = imputer.transform(pd.DataFrame(df[col]))
    return df

# Call function to replace/fill all missing values with median values
dengue_sj = ReplaceMissingValues(dengue_sj, col_with_null_sj, "median")
dengue_iq = ReplaceMissingValues(dengue_iq, col_with_null_iq, "median")

# Call function to again create and print table with count of Null values in all columns and check all are zero
NullColumnsTable_sj = CountMissingValues(dengue_sj, col_with_null_sj, "Count of Null/Missing Values for San Juan")
print(NullColumnsTable_sj)
NullColumnsTable_iq = CountMissingValues(dengue_iq, col_with_null_iq, "Count of Null/Missing Values for Iquitos")
print(NullColumnsTable_iq)


# **Note:** First, we do not want to simply drop the data because there is valuable information elsewhere and we need to test our prediction against the actual value when testing out model. Second, time-sensitive models depend on information from days prior. Dropping data would mess with time series models.

# ## Scale the Data 

# In[176]:


# Data Summary for San Juan
dengue_sj.describe(include=['float']).T


# In[178]:


# Data Summary for Iquitos
dengue_iq.describe(include=['float']).T


# In[179]:


# Select columns to be scaled
cols_to_scale = dengue_sj.select_dtypes(include=['float']).columns
cols_to_scale


# In[181]:


# Scale/Normalizing the data

# Function to scale/normalize certain columns using StandardScaler
def ScaleColumns(df, cols_to_scale):
    scaler = StandardScaler()
    for col in cols_to_scale:
        scaler=scaler.fit(pd.DataFrame(df[col]))
        df[col] = scaler.transform(pd.DataFrame(df[col]))
    return df

# Call the function to scale selected columns for each city dataset
dengue_sj = ScaleColumns(dengue_sj, cols_to_scale)
dengue_iq = ScaleColumns(dengue_iq, cols_to_scale)


# In[182]:


# Data Summary for San Juan
dengue_sj.describe(include=['float']).T


# In[183]:


# Data Summary for Iquitos
dengue_iq.describe(include=['float']).T


# In[200]:


# Visualize scaled data

# Function to plot istograms for all scaled columns in each city
def VizScaledCols(df, title):
    df[cols_to_scale].plot(kind="hist",figsize=(10,7))
    plt.title(title)
    plt.show()

# Call function to plot histograms for scaled columns in each city
VizScaledCols(dengue_sj, "Histograms for all scaled columns in San Juan")
VizScaledCols(dengue_iq, "Histograms for all scaled columns in Iquitos")


# # Explore and Visualize the features w.r.t Time

# ## Correlation

# In[204]:


# Set all Vegetation Index columns and Weather columns as features to be explored w.r.t time columns
features = cols_to_scale
features


# In[208]:


# Check correlation between features

# Function to check and print correlation between features 
def VisualizeCorrelation(df, title):
    # Create Correlation Matrix
    corr_matrix= df.corr()

    # Visualize correlation Matrix for all columns
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

VisualizeCorrelation(dengue_sj[features], 'Correlation Heatmap of Vegetation Index and Weather Features for San Juan')
VisualizeCorrelation(dengue_iq[features], 'Correlation Heatmap of Vegetation Index and Weather Features for Iqu')


# **Note:** 
# 
# * NVDI: The satellite imagery score of the vegetation growing in the city does not move together as much as I thought it would. This is especially true in San Juan. The North quadrants and the South quadrants have almost no relation to each other. There is a stronger correlation in Iquitos, but still not as strong as I had originally thought. This will make feature extraction difficult as I cannot simply combine the NVDI scores into one dimension.
# * Temperature: T The temperature variables (reanalysis_air_temp_k to reanalysis_min_air_temp_k) are strongly correlated together in San Juan, but not in Iquitos. Again, this makes feature extraction more difficult in Iquitos.
# 
# Both of these findings point to the need to create a unique model for each city.

# ## NDVI

# In[258]:


dengue_sj.columns[3:7]


# In[260]:


# create mean NVDI for each week for each city
dengue_sj['nvdi_mean'] = dengue_sj[dengue_sj.columns[3:7]].mean(axis = 1)
dengue_iq['nvdi_mean'] = dengue_iq[dengue_iq.columns[3:7]].mean(axis = 1)


# In[293]:


# Function to visualize the average of NDVI values for each week in each year compared to same computation for Mean NDVI
def VizNDVI(df, title):
    for i in df.columns[3:7]:
        df.groupby('weekofyear')[i].mean().plot(alpha = .3, figsize = (20, 5)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    df.groupby('weekofyear')['nvdi_mean'].mean().plot(alpha = 1, c = 'k', linewidth = 5)
    plt.title(title)
    plt.xlabel('Week of Year')
    plt.ylabel('NDVI')
    plt.show()

# Call the function
VizNDVI(dengue_sj,"NDVI Levels in San Juan")
VizNDVI(dengue_iq,"NDVI Levels in Iquitos")


# **Note:** 
# 
# San Juan: The NVDI scores in the Southwest and Southeast are consistently lower than the scores of the Northwest and Northeast quadrants. After averaging the score at each week over the 10 years of data, we can see that the NVDI score remains fairly constant. There is a minor dip in the last 4 weeks of the year (December) that could be interesting. In addition, we can see the impact of the front fill method for data imputation. In 1995, there were a few missing rows of NVDI data. The flat line comes from repeating the last known value over and over until an observation was recorded again.
# 
# Iquitos: * Iquitos shows a more homogenous NVDI score among its four quadrants. Visually, I can see an increase from Week 20 at 0.20 to Week 40 at 0.30 (May to October) in the NVDI scores .

# ## Weather

# In[310]:


# Visualize average of all weather features for each week in each year

def VizWeather(df, title):
    for i in df.columns[7:23]:
        df.groupby('weekofyear')[i].mean().plot(alpha = .3, figsize = (20, 7)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.ylabel('Normal Scale')
    plt.xlabel('Week of Year')
    plt.show()

VizWeather(dengue_sj,"Weather Features in San Juan")
VizWeather(dengue_iq,"Weather Features in Iquitos")


# **Note:** 
# 
# As we can see from the list of weather features, there are a couple of overlapping features that we are using. such as:
# 
# precipitation_amt_mm, reanalysis_precip_amt_kg_per_m2, reanalysis_sat_precip_amt_mm, and station_precip_mm all measure the rainfall in various ways
# reanalysis_air_temp_k, reanalysis_avg_temp_k, reanalysis_dew_point_temp_k, reanalysis_max_air_temp_k, reanalysis_min_air_temp_k, reanalysis_tdtr_k, station_avg_temp_c, station_diur_temp_rng_c, station_max_temp_c, and station_min_temp_c all measure various forms of the temperature.
# reanalysis_relative_humidity_percent and reanalysis_specific_humidity_g_per_kg measure the humidity.
# Each city follows a different trend (which makes sense, being in different hemispheres and climates).
# 
# San Juan: Hot, Wet and Humid. This time, it is in San Juan where we see an increase in all features from Week 20 to Week 40.
# 
# Iquitos: Variaion. The weather features do not move as tightly as San Juan does. Again, this was shown in the correlation plot above.

# # Explore and Visualize the target w.r.t Time

# ## Target summary

# In[305]:


dengue_sj[dengue_sj.columns[23]].describe()


# ## Target correlation with Vegetation Index and Weather Features

# In[309]:


# Checking correlation of target with other features
def VisualizeTargetCorrelation(df,title):
    # Create Correlation Matrix
    corr_matrix= df.corr(numeric_only=True)

    # Show Correlation for Total Cases
    plt.figure(figsize=(10, 4))
    corr_matrix['total_cases'].drop('total_cases').sort_values(ascending=False).plot.barh()
    plt.title(title)
    plt.show()


VisualizeTargetCorrelation(dengue_sj[dengue_sj.columns[3:24]], "Correlation of all features with target for San Juan")
VisualizeTargetCorrelation(dengue_iq[dengue_iq.columns[3:24]], "Correlation of all features with target for Iquitos")


# **Note:** all values for both city are between -0.2 to 0.3 which is very less, showing that these features have not much impact on target. now we look at target w. r. t time

# ## Target w.r.t Time

# In[308]:


# Visualize the Number of cases in each city over time

# Function to visualize the total cases in each city over time
def VizCasesInCity(df,title):
    df["total_cases"].plot(figsize = (10,4))
    plt.title(title)
    plt.xlabel("Year, Week of Year")
    plt.ylabel("Number of Cases")
    plt.show()

# Set year, weekofyear as index for each city split dataset
dengue_sj_target = dengue_sj.set_index(['year', 'weekofyear'])
dengue_iq_target = dengue_iq.set_index(['year', 'weekofyear'])

# Call the function to visualize the total cases in each city over time
VizCasesInCity(dengue_sj_target,"Number of Cases in San Juan")
VizCasesInCity(dengue_iq_target,"Number of Cases in Iquitos")


# In[307]:


# Visualize in details for each week
def VizCases(df, title):
    for i in set(df['year']):
        df2 = df[df['year'] == i]
        df2.set_index('weekofyear', drop = True, inplace = True)
        plt.plot(df2['total_cases'], alpha = .3)

    df.groupby('weekofyear')['total_cases'].mean().plot(c = 'k', figsize = (10,4))
    plt.legend(set(df['year']), loc='center left', bbox_to_anchor=(1, .5))
    plt.title(title)
    plt.xlabel('Week of the Year')
    plt.ylabel('Number of Cases')
    plt.show()

VizCases(dengue_sj, "Number of Cases per Week in San Juan")
VizCases(dengue_iq, "Number of Cases per Week in Iquitos")


# **Note:** 
# There are outbreaks in both Iquitos and San Juan toward the ends of each year. The increases in cases and outbreaks tend to happen in weeks 35 to 45 in San Juan and weeks 45 to 50 in Iquitos.

# # Create a Model

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




