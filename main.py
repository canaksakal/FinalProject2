import csv
import os

import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
from sklearn import linear_model
from sklearn.svm._libsvm import predict
from sklearn.utils import shuffle
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('covid19_by_country.csv')

# Dataset is too large so we must calculate numbers country by country
# We ask country alpha3code to user for calculation

dateCols = ['CountryAlpha3Code']

ulke = input('Enter the country code: ')

df = pd.read_csv("covid19_by_country.csv", parse_dates=dateCols)

data=df[(df['CountryAlpha3Code']  ==  ulke)]

# We must ignore null values

data = data.dropna() 

X = data.drop(['Country', 'CountryAlpha3Code', 'deaths', 'Date', 'deaths_PopPct',
               'confirmed_PopPct', 'DaysSince100Cases', 'DaysSince1Cases',
               'GRTStringencyIndex', 'recoveries_PopPct', 'recoveries_inc',
               'deaths_inc', 'confirmed_inc'], axis=1)
y = data[['deaths']]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

# Creating model
model = PassiveAggressiveClassifier(C=0.5, random_state=5, )

# Fitting model
model.fit(x_train, y_train)

exp_y  = y_test
y_pred = model.predict(x_test)

# Making prediction on test set
test_pred = model.predict(x_test)

print(f"Test Set Accuracy : {accuracy_score(y_test, test_pred) * 100} %\n\n")

print(f"Classification Report : \n\n{classification_report(y_test, test_pred)}")

print("R2 score\n")
print(metrics.r2_score(exp_y, y_pred))

print("Mean Squared Log Error\n")
print(metrics.mean_squared_log_error(exp_y, y_pred))

print("Explained Variance Score\n")
print(metrics.explained_variance_score(y_pred,y_test))

print("Mean Absolute Error\n")
print(metrics.mean_absolute_error(exp_y, y_pred))
