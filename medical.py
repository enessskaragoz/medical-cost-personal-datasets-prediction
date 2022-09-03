# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:57:46 2021

@author: Enes
"""
#ders6 : kutuphanelerin yuklenmesi
#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

