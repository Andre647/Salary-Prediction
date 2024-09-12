import os
import numpy as np
##pip install numpy=1.23.4
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    cross_val_score
)
from sklearn.preprocessing import OneHotEncoder

from scipy import stats

#Functions

@st.cache_data
def clean_and_modify_salary(salary):
    # Data Cleaning & Modifying
    salary = (
        salary
        .assign(
            Competitors=salary['Competitors'].fillna(0),
            Rating=salary['Rating'].fillna(salary['Rating'].mean()),
            Industry=salary['Industry'].fillna('Unknown'),
            Sector=salary['Sector'].fillna('Unknown'),
            age=salary['age'].fillna(salary['age'].mean()),
            Founded=salary['Founded'].fillna(salary['Founded'].mean())
        )
        .reset_index(drop=True)
    )

    salary.dropna(
        subset=['Revenue', 'Type of ownership', 'Size', 'Headquarters'],
        inplace=True
    )
    def seniority(title):
        title = title.lower()

        if any(keyword in title for keyword in
            ['sr', 'senior', 'lead', 'principal', 'manager', 'ii', 'iii']):
            return 'senior'
        elif any(keyword in title for keyword in
                ['iv', 'vp', 'director', 'chief']):
            return 'director'
        elif any(keyword in title for keyword in
                ['jr', 'i', 'junior']):
            return 'jr'
        else:
            return 'na'

    salary['seniority'] = salary['Job Title'].apply(seniority)

    def title_simplifier(title):
        title = title.lower()

        if 'data scientist' in title:
            return 'data scientist'
        elif 'data engineer' in title:
            return 'data engineer'
        elif 'analyst' in title:
            return 'analyst'
        elif 'machine learning' in title:
            return 'mle'
        elif 'manager' in title:
            return 'manager'
        elif 'director' in title:
            return 'director'
        else:
            return 'na'

    salary['Job Simp'] = salary['Job Title'].apply(title_simplifier)

    salary.rename(
        columns={
            'avg_salary': 'Salary',
            'company_txt': 'Company',
            'python_yn': 'python',
            'R_yn': 'R',
            'Job Simp': 'Job Title',
            'seniority' : 'Seniority'
        },
        inplace=True
    )

    # Cleaning categorical variables for our model
    salary = (
        salary
        .assign(
            Company=salary['Company'].apply(lambda x: x.split('\n')[0]),
            NCompetitors=salary['Competitors'].apply(
                lambda x: len(x.split(',')) if isinstance(x, str) else x
            ),
            Salary=salary.apply(
                lambda x: x['Salary'] * 2 if x['hourly'] == 1 else x['Salary'],
                axis=1
            )
        )
    )

    # Dropping the columns we are not using
    salary.drop(
        columns=[
            'Job Title',
            'Location',
            'Headquarters',
            'Industry',
            'Competitors',
            'hourly',
            'max_salary',
            'min_salary',
            'employer_provided',
            'Company Name',
            'Job Description',
            'Salary Estimate',
            'Founded',
            'age'
        ],
        inplace=True
    )

    # Changing column order for a cleaner view
    new_order = [
        'Type of ownership',
        'Seniority',
        'Size',
        'Sector',
        'Revenue',
        'NCompetitors',
        'Rating',
        'job_state',
        'python',
        'spark',
        'aws',
        'R',
        'excel',
        'Salary'
    ]
    
    return salary[new_order]

@st.cache_resource
def preprocess_and_train_model(df):
    # Remover colunas indesejadas
    df = df.drop(columns=[ 'R', 'excel'])
    
    # Calcular Z-Scores e remover outliers
    z_scores = np.abs(stats.zscore(df.select_dtypes('number')))
    limit = 4.0
    df = df[(z_scores < limit).all(axis=1)]

    # One-Hot Encoding para variáveis categóricas
    hot_encode = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    categorical_features = df.select_dtypes(exclude='number')
    hot_encode.fit(categorical_features)
    
    df_hot = pd.DataFrame(
        hot_encode.transform(categorical_features),
        columns=hot_encode.get_feature_names_out()
    )
    
    # Combinar variáveis categóricas codificadas com as numéricas
    df = df_hot.join(df.select_dtypes(include='number'))
    df = df.dropna()

    # Separar features (X) e target (y)
    X = df.drop(columns=['Salary'])
    y = df['Salary']

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar RandomForestRegressor
    rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)

    # Definir parâmetros para GridSearchCV
    parameters = {
        'n_estimators': range(10, 300, 50),
        'criterion': ['friedman_mse', 'absolute_error'],
        'max_features': [None, 'sqrt', 'log2']
    }

    # Aplicar GridSearchCV
    gs = GridSearchCV(
        estimator=rf,
        param_grid=parameters,
        error_score='raise',
        cv=3
    )
    
    gs.fit(X_train, y_train)

    # Retornar o modelo treinado e os melhores parâmetros encontrados
    return gs.best_estimator_, X_test, y_test

st.write('# Introduction')

"In this Data Science project, our goal is to develop a predictive model to estimate a data jobs position's salary based on various characteristics such as State, Rating of the company, important skills and seniority. Using regression techniques, we will see how those variables influences compensation. The project involves exploratory data analysis, the construction of different regression models, and the evaluation of their performance, aiming to create an accurate tool for predicting salaries in various contexts."
st.divider()

st.write("# Dataframes")

URL = (
    'https://raw.githubusercontent.com/Andre647/Salary-Prediction/main/data/salary_data_cleaned.csv'
)
salary = pd.read_csv(URL, na_values=-1)

if st.checkbox('Initial dataframe'): 
    st.dataframe(salary, use_container_width=True)
    salary.shape

df = clean_and_modify_salary(salary)

if st.checkbox('Final dataframe'): 
    st.dataframe(df, use_container_width=True)
    df.shape

st.divider()

st.write("# EDA")
sns.set_palette(palette='viridis', n_colors=1)

"My initial idea was to navigate through the SweetViz report and visualize how all the variables relate to the TARGET. However, due to a Numpy update, it is not possible to generate the object."
"Feel free to view it downloading the file Salary_EDA___Prediction.ipynb in my github page https://github.com/Andre647/Salary-Prediction/blob/main/Salary_EDA___Prediction.ipynb"


colunas = ["Skills", "Seniority", "Size", "Type of ownership", "Sector", "Revenue", "NCompetitors", "Rating"]
colunas_boxplot = ["Seniority", "NCompetitors"]


plot_type = st.selectbox("Select plot", colunas)

fig, ax = plt.subplots(figsize=(10, 5))

if plot_type in colunas_boxplot:
    st.write(f"## {plot_type}")
    sns.boxplot(x=plot_type, y='Salary', data=df, ax=ax)

elif plot_type in "Skills":
    st.write(f"## {plot_type}")
    fig, ax = plt.subplots(1,4, figsize=(10,5), sharey=True)
    for ax, coluna in zip(ax.flatten(), ['python', 'aws', 'excel', 'spark']):
        sns.boxplot(x=coluna, y='Salary', data=df, ax=ax)

elif plot_type in "Rating":
    st.write(f"## {plot_type}")
    sns.scatterplot(x=plot_type, y='Salary', data=df, ax=ax)

else:
    st.write(f"## {plot_type}(mean)")
    sns.barplot(x='Salary', y=plot_type, data=df, ax=ax)

# Exibir o gráfico no Streamlit
st.pyplot(fig)

"""
  * It is possible to see that, as expected, the **seniority** of the position has a significant impact on the salary it offers. However, more than half of our data does not have this defined. Standardizing seniority for each position on Glassdoor would help us address this issue.

  * Digging deeper, we can also see that positions place more value on individuals with expertise in **Python, AWS, and even Spark**, which was anticipated, given that our positions are mostly in the Data field. We can also observe that Excel is becoming undervalued in the market.

  * The report also provides other relevant information, such as **California** being one of the highest-paying states and that companies with **higher ratings** also tend to offer **higher salaries**.

"""

st.divider()

clf, X_test, y_test = preprocess_and_train_model(df)
st.write("# Results ")

st.write("## Trained Model")
clf

y_pred = clf.predict(X_test)
r2 = round(r2_score(y_test, y_pred),4)
st.write(f"## R² {r2}")

"We will start visualizing the residuals, which helps us identify whether the model is biased or if it is 'randomly making errors,' which is ideal."
residuals = y_test - y_pred

fig = plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Values (Predictions)')
plt.ylabel('Residuals')
plt.title('Resisduals vs. Values')
st.pyplot(fig)

"Next, we will analyze whether the residuals follow a normal distribution. This is important because the normality of the residuals helps validate the model, ensures confidence in the inferences it provides, and guarantees that there are no significant uncaptured patterns, contributing to the model's robustness and reliability."
fig, ax= plt.subplots(1,2, figsize=(10, 6))

sns.histplot(residuals, kde=True, bins=30, ax=ax[0])
ax[0].set_xlabel('Residuals')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Residuals Histogram')

stats.probplot(residuals, dist="norm", plot=ax[1])
plt.title('Residuals QQ-Plot')

plt.tight_layout()
st.pyplot(fig)

st.write("## Conclusion")
"""
## R-squared

An R-squared of 0.46 in a regression model indicates that approximately 46% of the variability in the outcome (or dependent variable) is explained by the model. In other words, about 46% of the variation in the data can be attributed to the independent variables included in the model, while the remaining 54% is due to other factors not included in the model or random variations.

* Moderate Explanation: The model explains a moderate amount of the variability in the data. While it is not a perfect model, it provides a significant amount of explanation for the outcome.

* Room for Improvement: An R-squared of 0.47 suggests that there is room to improve the model. It may be useful to explore other variables or adjust the model to enhance the explanation of the variability.
"""
"""
## Residuals
* The Residuals vs. Predicted plot shows that the errors are randomly distributed, indicating that our model does not exhibit any 'bias' or tendency toward a certain value.

* Our histogram of residuals and the QQ-Plot indicate that they are following a normal distribution, validating the results of the model.
"""
