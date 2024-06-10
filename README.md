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

