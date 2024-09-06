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

xx_err = [xx - xx_mean for xx in main_df['EXP']]
yy_err = [yy - yy_mean for yy in main_df['SALARY']]

errors = [xer * yer for xer, yer in zip(xx_err,yy_err)]
pred_slope = sum(errors)/sum([x*x for x in xx_err])

pred_intercept = yy_mean - pred_slope * xx_mean

RSS_values = [(yy-pred_intercept-pred_slope*xx)**2 for xx, yy in zip(main_df['EXP'],main_df['SALARY'])] 

RSS = sum(RSS_values)


RSE = np.sqrt(RSS/(len(RSS_values)-2))

TSS = sum([(yy - yy_mean)**2 for yy in main_df['SALARY']])

R_squared = 1 - RSS/TSS

std_dev = main_df.std()

exp_stddev = std_dev[0]
salary_stddev = std_dev[1]