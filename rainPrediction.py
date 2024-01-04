'''
Project by -> Thomas Haskell

TOPIC -> Rain Prediction in Australia
SOURCE -> IBM Machine Learning with Python Certification Course

DESCRIPTION -> Using multiple calssification algorithms learned in the course, I will 
train and test a model using new knowledge of evaluation metrics.
    Specifically, these algorithms will be used:
        - Linear Regression
        - KNN
        - Decision Trees
        - Logistic Regression
        - SVM
    Which will be evaluated with these metrics:
        - Accuracy Score
        - Jaccard Index
        - F1-Score
        - LogLoss
        - Mean Absolute Error
        - Mean Squared Error
        - R2-Score

OBJECTIVES:
1. Develop a classification model using Logistic Regression Algorithms
2. Visualize data and gain familiarity with pandas and numpy libraries
'''
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

## Importing Libraries ##
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics


## Downloading the dataset ##
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  #workaround for SSL certificate verification
def download(url, filename):
    urllib.request.urlretrieve(url, filename)
    print("Download Complete")
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'
download(path, "Weather_Data.csv")
filename = "Weather_Data.csv"
df = pd.read_csv(filename)
df.head()


## Pre-Processing  ##
# performing "one hot encoding" to convert categorical variables to numerical variables
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
# converting target column 'RainTomorrow'
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)


## Train/Test Splitting ##
df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1) # setting feature variable
Y = df_sydney_processed['RainTomorrow'] # setting target variable
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)


#### Linear Regression Modeling ####
LinearReg = LinearRegression() # creates the model
x = np.asanyarray(x_train)
y = np.asanyarray(y_train)
LinearReg.fit(x,y) # trains the model
predictions = LinearReg.predict(x_test) # predicts on test set
# Evaluating results
LinearRegression_MAE = np.mean(np.abs(predictions - y_test))
LinearRegression_MSE = np.mean((predictions - y_test) ** 2)
LinearRegression_R2 = metrics.r2_score(y_test, predictions)
print("Linear Regression, Mean Absolute Error: ", LinearRegression_MAE)
print("Linear Regression, Mean Squared Error: ", LinearRegression_MSE)
print("Linear Regression, R2 Score: ", LinearRegression_R2)
# Showing results in tabular format as a data frame
Report = pd.DataFrame({'Model' : ['Linear Regression'], 'MAE': [LinearRegression_MAE], 'MSE': [LinearRegression_MSE], 'R2': [LinearRegression_R2]})
print(Report)