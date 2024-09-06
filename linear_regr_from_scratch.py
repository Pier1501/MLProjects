# Machine Learning Project - Linear Regression from scratch in Python

# This project is based on examples taken from the MIT project-based-learning open-source GitHub repository
# https://github.com/practical-tutorials/project-based-learning


# The dataset is taken from Kaggle
# https://www.kaggle.com/datasets/saquib7hussain/experience-salary-dataset

import numpy as np
import pandas as pd
import matplotlib as plt


original_df = pd.read_csv('Experience-Salary.csv')

original_df.describe
original_df_df_shape = original_df.shape

cols_index = {'exp(in months)':'EXP', 'salary(in thousands)': 'SALARY'}
main_df=original_df.rename(columns=cols_index)

# Months of experience is the indipendent variable x
# Salary in thousands of dollars is the dependant variable y

# We find the means

xx_mean = main_df['EXP'].mean()
yy_mean = main_df['SALARY'].mean()

# We find the medians

xx_median = main_df['EXP'].median()
yy_median = main_df['SALARY'].median()



# We are predicting a simple linear regression so there will be a linear relationship in the form
# Y = Bx + Q
# where Q is the intercept and B is the slope and are going to be the predicted parameters
# In our case the result is SALARY = B * EXP + Q

# Calculating all errors for the variables
xx_err = [xx - xx_mean for xx in main_df['EXP']]
yy_err = [yy - yy_mean for yy in main_df['SALARY']]

errors = [xer * yer for xer, yer in zip(xx_err,yy_err)]

# Finding the predicted values for slope and intercept
pred_slope = sum(errors)/sum([x*x for x in xx_err])

pred_intercept = yy_mean - pred_slope * xx_mean

# Finding the RSS, RSE, TSS values
RSS_values = [(yy-pred_intercept-pred_slope*xx)**2 for xx, yy in zip(main_df['EXP'],main_df['SALARY'])] 

RSS = sum(RSS_values)


RSE = np.sqrt(RSS/(len(RSS_values)-2))

TSS = sum([(yy - yy_mean)**2 for yy in main_df['SALARY']])

# Evaluating the R_squared statistic
R_squared = 1 - RSS/TSS

# Now we are going to do an inference on the predicted values, to make sure 
# we have reliable values
# We are creating a table with the coefficients, std error, t-statistic and p-value

y_std_dev = main_df.std()[1]

xx_sum_sq_err = sum([(xx-xx_mean)**2 for xx in main_df['EXP']])

slope_std_err_sq = (y_std_dev*y_std_dev)/xx_sum_sq_err

intercept_std_err_sq = (y_std_dev*y_std_dev)*(1/len(main_df['EXP'])+ xx_mean/xx_sum_sq_err)

slope_t_stat = 
intercept_t_stat = 