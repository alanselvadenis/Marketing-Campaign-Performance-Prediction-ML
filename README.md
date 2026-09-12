# Marketing-Campaign-Performance-Prediction-ML-

Problem Statement:
Marketing teams across multiple brands such as Nykaa, Purplle, and Tira generate large volumes of campaign data, including impressions, clicks, conversion rates, acquisition costs, revenue, and ROI. However, this data is often stored in raw and complex formats like CSV files, making it difficult to analyze and extract meaningful insights.
Additionally, challenges such as missing values, inconsistent ROI calculations, and multiple marketing channels within a single column increase the complexity of data preprocessing and impact the reliability of analysis.
The objective is to transform this raw data into a structured format through data cleaning, preprocessing, and exploratory data analysis (EDA). This includes handling missing values, applying multi-label encoding for categorical features, and creating new features such as profit or loss indicators based on ROI.
This project aims to build a complete end-to-end machine learning pipeline that enables candidates to:
Clean and preprocess marketing campaign data
Handle missing values and perform data transformation
Apply multi-label encoding for features like marketing channels
Perform exploratory data analysis (EDA)
Engineer new features such as Profit/Loss classification
Build regression models to predict revenue
Develop classification models to predict campaign success (profit vs loss)
Evaluate model performance using appropriate metrics
Objective:
The objective of this project is to design and implement a Marketing Campaign Performance Prediction System that follows the complete machine learning lifecycle:
Data Collection
Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering (Multi-label Encoding, Profit/Loss Creation)
Model Building (Regression & Classification)
Model Evaluation and Performance Analysis

Approach:
1. Data Collection
Import the dataset in CSV format
Load the dataset into Python using Pandas DataFrame
Understand the structure, columns, and data types



2. Data Preprocessing
Handle missing values (null data)
Remove duplicate records
Perform data type conversion
Clean and standardize the dataset
Validate and correct ROI values (if required)

3. Feature Engineering
Create new feature: Profit/Loss flag based on ROI
Apply multi-label encoding for Channel_Used column
Select relevant features for modeling
Perform feature transformation if needed

4. Exploratory Data Analysis (EDA)
Analyze campaign performance across brands
Identify top-performing and low-performing campaigns
Explore relationships between spend, clicks, revenue, and ROI
Analyze channel-wise effectiveness

5. Model Building
Split the dataset into training and testing sets to evaluate model performance
Build regression models to predict Revenue based on campaign features
Build classification models to predict Profit/Loss using the engineered Profit_Flag variable
Train models using appropriate algorithms such as Linear Regression, Logistic Regression, Decision Tree, and Random Forest
Perform feature selection and ensure no data leakage (exclude ROI from classification features)
Tune model parameters to improve performance
Validate models using unseen test data

6. Model Evaluation
Evaluate regression models using RMSE, MAE, R², MSE (R² >= 0.95)
Evaluate classification models using Accuracy, Precision, Recall, F1-score (Accuracy >= 0.95)
Compare model performance and select the best model
7. Insights & Reporting
Generate insights on campaign performance
Identify key factors affecting profitability
Provide data-driven recommendations for marketing strategies
Support decision-making using model predictions
8. Application Development
Build an interactive user interface using Streamlit to:
Input campaign details (e.g., impressions, clicks, spend, channels, etc.)
Apply the same preprocessing and feature engineering steps (encoding, transformations)
Display predicted Revenue (Regression Output)
Display predicted Profit/Loss (Classification Output)
Visualize key inputs and prediction results for better understanding
Dataset Description:
The dataset contains detailed information about marketing campaigns, customer engagement, and financial performance:
Campaign_ID: Unique identifier for each campaign.
Campaign_Type: Type of marketing campaign (e.g., awareness, conversion, promotion).
Target_Audience: Group of customers targeted by the campaign.
Duration: Total duration of the campaign (in days).
Channel_Used: Marketing channels used for the campaign (may contain multiple channels).
Impressions: Number of times the campaign was displayed to users.
Clicks: Number of users who clicked on the campaign.
Leads: Number of potential customers generated.
Conversions: Number of successful actions (e.g., purchases or sign-ups).
Revenue: Total revenue generated from the campaign.
Acquisition_Cost: Cost incurred to run the campaign.
ROI: Return on Investment indicating campaign profitability.
Language: Language used in the campaign content.
Engagement_Score: Measure of user interaction and engagement level.
Customer_Segment: Category of customers targeted (e.g., new, returning, premium).
Date: Date on which the campaign was executed.
