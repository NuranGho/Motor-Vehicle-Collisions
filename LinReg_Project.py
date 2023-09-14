### Import packages in the virtual environment
import pandas as pd   ## search the package "pandans"
import statsmodels.api as sm    ## search the package "statsmodels"
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scipy
from scipy.stats import shapiro
import sys
import os
os.chdir(r"C:\Users\nuran\PycharmProjects\CIS9660\Project")
pd.set_option('display.max_columns', 500)
## Display all columns when printing results

mydata = pd.read_csv("vehicle_collisions.csv")
## Remember to include “r” first inside the bracket.

file = open('LinearReg_OutputTime.txt','wt')
sys.stdout = file
## Open a text file and save all results into the text file

#print(mydata.describe())
## Display summary statistics of the data

df = pd.DataFrame(mydata)
##Transform the dataset into two-dimensional, size-mutable, potentially heterogeneous tabular data

######################## Linear regression with only one independent variable
x = df['CONTRIBUTING FACTOR VEHICLE 1_Driver Inattention/Distraction']
##Define the independent variables in the model
y = df['NUMBER OF PERSONS KILLED']
##Define the dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model1 = sm.OLS(y, x).fit()
##Fit the model
results_model1 = model1.summary()
print(results_model1)
##Output the results

df["predicted1"] = model1.predict(x)
##Caculate the predicted value of dependent variable based on the model

sns.scatterplot(data=df, x="CONTRIBUTING FACTOR VEHICLE 1_Driver Inattention/Distraction", y="NUMBER OF PERSONS KILLED")
plt.show()
##Plot the predicted value against the actual value of charges.



######################## Linear regression with only multiple independent variables

x = df[['CRASH TIME','NUMBER OF PERSONS INJURED', 'CONTRIBUTING FACTOR VEHICLE 1_Driver Inattention/Distraction', 'CONTRIBUTING FACTOR VEHICLE 1_Unspecified', 'CONTRIBUTING FACTOR VEHICLE 1_Following Too Closely',
        'CONTRIBUTING FACTOR VEHICLE 1_Failure to Yield Right-of-Way', 'CONTRIBUTING FACTOR VEHICLE 1_Passing or Lane Usage Improper', 'CONTRIBUTING FACTOR VEHICLE 1_Backing Unsafely',
        'VEHICLE TYPE CODE 1_SEDAN', 'VEHICLE TYPE CODE 1_Station Wagon/Sport Utility Vehicle', 'VEHICLE TYPE CODE 1_TLC', 'VEHICLE TYPE CODE 1_Pick-up Truck',
        'VEHICLE TYPE CODE 1_BOX TRUCK', 'VEHICLE TYPE CODE 1_Bike']]
##Define the independent variables in the model
y = df['NUMBER OF PERSONS KILLED']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model2 = sm.OLS(y, x).fit()
##Fit the model
results_model2 = model2.summary()
print(results_model2)
##Output the results

df["predicted2"] = model2.predict(x)
##Caculate the predicted value of dependent variable based on the model

sns.scatterplot(data=df, x="CONTRIBUTING FACTOR VEHICLE 1_Driver Inattention/Distraction", y="NUMBER OF PERSONS KILLED")
plt.show()
##Plot the predicted value against the actual value of charges.

##Regression diagnostic I: multicollinearity
corr_matrix = df.corr()
print(corr_matrix)

##Regression diagnostic 2: heteroscedasticity in the error term
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(model2.resid, model2.model.exog)
print(lzip(names, test))

##Regression diagnostic 3: autocorrelation
##This is not a time-series dataset. Thus, the concern about autocorrelation is minimul.

##Regression diagnostic 4: model specification errors
##1)The model does not exclude any “core” variables.
##2)The model does not include superfluous variables.
##3)The functional form of the model is suitably chosen.
##4)There are no errors of measurement in the regressand and regressors.
##5)Outliers in the data, if any, are taken into account.

print(mydata.describe())

##6)The probability distribution of the error term is well specified.
residuals = model2.resid
print(shapiro(residuals))

file.close()
##Close the text file.


