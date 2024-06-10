# singapore-flat-price-resale-prediction

# Introdcution
The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.


## Technologies used
Data Collection, Data Wrangling, Data Preprocessing, EDA, Model Building, Model Deployment
## Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from streamlit_option_menu import option_menu
import jobolib
## Work flow of the project
## Workflow Overview
1.Data Collection
2.Data Preprocessing
3.Exploratory Data Analysis (EDA)
4.Feature Engineering
5.Model Building
6.Model Evaluation
7.Deployment

Detailed Steps
1. Data Collection
  Source: Gather data from publicly available sources such as Singapore's HDB resale flat prices dataset from data.gov.sg.
  Tools: Python libraries like pandas or web scraping tools if necessary.

2. Data Preprocessing
    Handling Missing Values: Identify and handle missing values.
    Data Types: Ensure data types are correct (e.g., dates should be in datetime format).
    Categorical Data: Convert categorical data into numerical formats if necessary.

3. Exploratory Data Analysis (EDA)
    Descriptive Statistics: Summary statistics of the data.
    Visualization: Use plots to visualize relationships (e.g., price distributions, trends over time).

4. Feature Engineering
    Create New Features: Based on existing data (e.g., age of flat,years holding lower storey and upper storey).
    Encode Categorical Variables: Use ordinal encoding.

5.Model Building:
    choose Model: Choosing a best algorithm i get Random forest as best algorithm due ti pickle file size am using decision tree.
    Train-Test Split: Split data into training and testing sets.


Creating a workflow for analyzing Singapore flat resale prices involves several key steps: data collection, data preprocessing, exploratory data analysis (EDA), feature engineering, model building, model evaluation, and deployment. Here’s a comprehensive workflow to guide you through this process:

Workflow Overview
Data Collection
Data Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Model Building
Model Evaluation
Deployment
Detailed Steps
1. Data Collection
Source: Gather data from publicly available sources such as Singapore's HDB resale flat prices dataset from data.gov.sg.
Tools: Python libraries like pandas or web scraping tools if necessary.
python
Copy code
import pandas as pd

# Example: Loading data from a CSV file
data_url = "https://data.gov.sg/dataset/hdb-resale-prices/resource/1b702208-44bf-4829-b620-4615ee19b57c"
data = pd.read_csv(data_url)
2. Data Preprocessing
Handling Missing Values: Identify and handle missing values.
Data Types: Ensure data types are correct (e.g., dates should be in datetime format).
Categorical Data: Convert categorical data into numerical formats if necessary.
python
Copy code
# Example: Handling missing values and data types
data.dropna(inplace=True)
data['month'] = pd.to_datetime(data['month'])
3. Exploratory Data Analysis (EDA)
Descriptive Statistics: Summary statistics of the data.
Visualization: Use plots to visualize relationships (e.g., price distributions, trends over time).
python
Copy code
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Plotting resale prices over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='month', y='resale_price')
plt.title('Resale Prices Over Time')
plt.xlabel('Month')
plt.ylabel('Resale Price')
plt.show()
4. Feature Engineering
Create New Features: Based on existing data (e.g., age of flat, proximity to amenities).
Encode Categorical Variables: Use one-hot encoding or label encoding.
python
Copy code
# Example: Creating new features
data['flat_age'] = data['month'].dt.year - data['lease_commence_date']

# One-hot encoding for flat types
data = pd.get_dummies(data, columns=['flat_type'])
5. Model Building
Choose Model: Select appropriate models (e.g., linear regression, random forest, gradient boosting).
Train-Test Split: Split data into training and testing sets.

6. Model Evaluation
Metrics: Evaluate model performance using metrics like RMSE, MAE, R².
Cross-Validation: Ensure robustness of the model.

7. Deployment
Save Model: Serialize the model using joblib or pickle.
Deployment:Deployed in Render.
## Deployment

https://singapore-flat-price-resale-prediction.onrender.com


## Demo

Insert gif or link to demo
