#Multiple linear Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #used to create the matrix of the variable remove the last column
y = dataset.iloc[:, 4].values    # select the last column


#we have 1 categorical variable in the dataframe so we encode the variable to dummy variables
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3]) #create dummy variables
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable Trap
X = X[:,1:] #remove first column from x also can be done with library



# Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
#stats.model api library 
import statsmodels.formula.api as sm #this doesn't work
import statsmodels.regression.linear_model as lm #use this

X = np.append(arr = np.ones((50,1)).astype(int),values = X,axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
#create a new regresser which will be object of the class
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#remove index 2 with the highest p value
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

































