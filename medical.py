# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:57:46 2021

@author: Enes
"""

#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

#2.VERİ ÖNİŞLEME
#2.1 VERİ YÜKLEME

df = pd.read_csv("insurance.csv")
print(df)

# Data Analysis
print(df.info())
print(df.head(10))
print(df.isna().sum())    # checks for missing data
print(df.bmi.describe())    # Generate descriptive statistics.


# Relationship between "smoker" and "charges"
sns.set_style("darkgrid")

ax = sns.scatterplot(x= "smoker",
           y= "charges",
           hue= "smoker",     
           data= df)
ax.set(xlabel=None)
plt.show()


smoker = df[df["smoker"] == "yes"]
nonsmoker = df[df["smoker"] == "no"]
# Relationship between "smoker" and "region"
plot = sns.catplot(x= "region",
           data= smoker,
           kind= "count")
plot.set(xlabel= "Region",
    ylabel= "Smoker")

plt.show()


# Relationship between "bmi" and "sex"
sns.set_style("darkgrid")

ax = sns.scatterplot(x= "sex",
           y= "bmi",
           hue= "sex",     
           data= df)
ax.set(xlabel=None)
plt.show()


# Which "Region" has the most number of "Childeren"
print(df.groupby(["region"])[["children"]].agg([np.sum]))


# Relationship between "age" and "bmi"
sns.lineplot(x = "age",
             y = "bmi",
             data = df)
plt.show()


# Relationship between "bmi" and "children"
sns.lineplot(x= "children",
             y= "bmi",
            data= df)
plt.show()



# Is there any outlier in the feature of "BMI"?
sns.relplot(x = "bmi",
              y= "sex",
            data = df)

bmi_out = df[df["bmi"] >= 40]
bmi_out_2 = df[df["bmi"] <= 10]

sns.relplot(x = "bmi",
              y= "sex",
            data = bmi_out)
sns.relplot(x = "bmi",
              y= "sex",
            data = bmi_out_2)
plt.show()


# Relationship between "BMI" and "Charges"
insurance_charge = df[df["charges"] > 10000]
sns.scatterplot(x= "bmi",
             y= "charges",
            data= insurance_charge)
plt.show()


# "region", "smoker" and "bmi"
plt.figure(figsize = (17, 8))
sns.barplot(x = 'region', y = 'bmi', hue = 'smoker', data = df)
plt.show()



# Data Preprocessing
# Converting categorical data to numeric data
# label encoder

def label_encoding(column_name):
    le = LabelEncoder()
    df[column_name] = le.fit_transform(df[column_name])
    
label_encoding("smoker")
label_encoding("sex")


# One-Hot Encoder
ohe = pd.get_dummies(df["region"])    # Convert categorical variable into dummy/indicator variables.    

# ohe and df concat
df = pd.concat([df, ohe], axis = 1)
df.drop("region",axis = 1, inplace = True)


# Splitting Dataset
X = df.drop('charges',axis=1)
y = df['charges']


# Feature Scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X[0:5]    # As a result of this operation, all data takes a value between 0 and 1.

# Training and test data of dependent and independent variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Model Selection
# Models
linreg = LinearRegression()
dectree = DecisionTreeRegressor()
randforest = RandomForestRegressor()
svm = SVR()


# Score comparison with cross validation
# LinearRegression scores
linreg_scores = cross_val_score(linreg, X_train, y_train, scoring = "neg_mean_squared_error", cv = 10)

# Decision Tree scores
dectree_scores = cross_val_score(dectree, X_train, y_train, scoring = "neg_mean_squared_error", cv = 10)

# Random Forest scores
randforest_scores = cross_val_score(randforest, X_train, y_train, scoring = "neg_mean_squared_error", cv = 10)

# SVM scores
svm_scores = cross_val_score(svm, X_train, y_train, scoring = "neg_mean_squared_error", cv = 10)


# Function displaying Regression Evaluation Metrics
def score_display(scores):
    scores = np.sqrt(-scores)
    print(f"""
          RMSE Scores : {scores}
          Mean : {scores.mean()}
          Standart Deviation : {scores.std()}
          """)
          
score_display(linreg_scores)  
score_display(dectree_scores)
score_display(randforest_scores)
score_display(svm_scores)
# RandomForestRegressor has the lowest RMSE(Root Mean Squared Error). Therefore, we will continue with that.

        
# Parameter Tuning
parameters = {'n_estimators': [3, 10, 20, 50], 
          'n_jobs': [2, 3, 4, 10]} 
# Add more parameters and try that way
grid_s = GridSearchCV(randforest, parameters, 
                     cv=5,
                     scoring='neg_mean_squared_error')

grid_s.fit(X_train,y_train)
print(grid_s.best_params_)


# To see the Root mean squared Error and the parameters 
for mean_score,params in zip((grid_s.cv_results_['mean_test_score']),
                             (grid_s.cv_results_['params'])):
    print(np.sqrt(-mean_score),'    ',params);
    
    
# Predicting
predictions = grid_s.best_estimator_.predict(X_test)
y_test[0:10].values

comparison = pd.DataFrame({'Y Test': y_test[0:10].values,
                          'Predictions' : predictions[0:10]})


# Evaluation
# In this code we will use r-squared, Mean Squared Error(MSE), Mean Absolute Error(MAE).
def regression_evaluation(preds):
    mse = mean_squared_error(y_test,preds)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test,preds)
    mae = mean_absolute_error(y_test,preds)
    
    print(f"Mean Absolute Error: {mae} \nMean Squared Error:{mse} \nRoot Mean Squared Error:{rmse} \nR Squared Value:{r_squared}")

print(regression_evaluation(predictions))


# Finding the Confidence Interval Of %95
"""
Confidence intervals are intervals in which we have a certain confidence to 
find the real value of the observable we measure. Scientists usually search 
for the 95% confidence interval.
"""

from scipy import stats

confidence = 0.95
squared_errors = (predictions - y_test) ** 2

print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
loc=squared_errors.mean(),
scale = stats.sem(squared_errors))))



