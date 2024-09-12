# Salary Prediction App

This project uses machine learning techniques to predict salaries for data-related jobs based on several factors such as company rating, state, required skills, and seniority level. The goal is to create a tool to help professionals and employers estimate compensation for specific job positions.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
In this Data Science project, we aim to develop a predictive model that estimates a data job's salary based on various features like company rating, skills, state, and seniority. We explore these features, build regression models, and evaluate their performance to create a reliable salary prediction tool.

## Data
The dataset used in this project was cleaned and transformed to fit the model. You can explore the original data by downloading it from the following link:

- [Salary Data](https://raw.githubusercontent.com/Andre647/Salary-Prediction/main/data/salary_data_cleaned.csv)

## Data Cleaning
Several transformations were applied to the dataset to clean and format it properly for analysis. Key operations include:

- Filling missing values for competitors, ratings, industry, sector, and company age.
- Simplifying job titles and creating a new column for seniority levels.
- Cleaning categorical variables such as the company name and the number of competitors.
- Dropping unnecessary columns like company description.
- Scaling salaries by adjusting for hourly rates.

## Exploratory Data Analysis (EDA)
Initial exploration using SweetViz showed insights, such as the impact of seniority, state, and skills (Python, AWS, Spark, etc.) on salary.

Unfortunately, due to a Numpy update, SweetViz reports are not directly viewable in the app. However, you can still check the EDA results by downloading the notebook from the following link:

- [EDA & Prediction Notebook](https://github.com/Andre647/Salary-Prediction/blob/main/Salary_EDA___Prediction.ipynb)

## Model Training
The dataset was split into training and test sets, and a RandomForestRegressor was trained. A GridSearchCV was applied to optimize the model's hyperparameters.

Steps:
1. Data was encoded using OneHotEncoding for categorical variables.
2. The dataset was cleaned of outliers using Z-scores.
3. The best model was selected using cross-validation.
4. Final evaluation metrics were calculated on the test set.

## Results
The model achieved an R² score of **0.46**, meaning that 46% of the variance in salary can be explained by the independent variables in the model. 

### Residual Analysis
The residuals plot and distribution show that the model's errors are randomly distributed and normally distributed, confirming that the model has no bias and is valid for predictions.

## Conclusion
While the model explains a moderate amount of variability (46%) in salary, there is still room for improvement by introducing new features or adjusting the model. The residual analysis also confirms the model's reliability, as the errors follow a normal distribution.


