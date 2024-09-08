# Machine Learning Project - Linear Regression from scratch in Python

# This project is based on examples taken from the MIT project-based-learning open-source GitHub repository
# https://github.com/practical-tutorials/project-based-learning


# The dataset is taken from Kaggle
# https://www.kaggle.com/datasets/saquib7hussain/experience-salary-dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import sklearn.linear_model as sklin


original_df = pd.read_csv('Experience-Salary.csv')

original_df.describe
original_df_df_shape = original_df.shape

cols_index = {'exp(in months)':'EXP', 'salary(in thousands)': 'SALARY'}
main_df=original_df.rename(columns=cols_index)


# Months of experience is the indipendent variable x
# Salary in thousands of dollars is the dependant variable y

# Let's see the scatterplot of the values
plt.scatter(main_df['EXP'], main_df['SALARY'], s=3)

# We find the means

xx_mean = main_df['EXP'].mean()
yy_mean = main_df['SALARY'].mean()

# We find the medians

xx_median = main_df['EXP'].median()
yy_median = main_df['SALARY'].median()

# Let's compare the values

mean_med_table = pd.DataFrame({'Mean': [xx_mean,yy_mean], 'Median': [xx_median,yy_median]}, index = ['Experience', 'Salary'])

# The two distribution are almost normal, since median and mean are almost equal


# Let's see the distribution of the columns with boxplots

fig = plt.figure(figsize = (10, 10))
ax = fig.add_axes([0, 0, 1, 1])

exp_distr = ax.boxplot(main_df['EXP'])

figg = plt.figure(figsize = (10, 10))
axx = figg.add_axes([0, 0, 1, 1])

salary_distr = axx.boxplot(main_df['SALARY'])


# We are predicting a simple linear regression so there will be a linear relationship in the form
# Y = Bx + Q
# where Q is the intercept and B is the slope and are going to be the predicted parameters
# In our case the result is SALARY = B * EXP + Q

# Calculating all errors for the variables
xx_err = [x - xx_mean for x in main_df['EXP']]
yy_err = [y - yy_mean for y in main_df['SALARY']]

errors = [xer * yer for xer, yer in zip(xx_err, yy_err)]

# Finding the predicted values for slope and intercept
pred_slope = sum(errors)/sum([x*x for x in xx_err])

pred_intercept = yy_mean - pred_slope * xx_mean

prediction_table = pd.DataFrame({'Coefficient': [pred_slope, pred_intercept]}, index = ['Slope', 'Intercept'])

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

var_x = sum([x**2 for x in xx_err])
var_y = sum([y**2 for y in yy_err])

std_err_slope = np.sqrt((RSE**2)/var_x)

std_err_intercept = np.sqrt((RSE**2/len(xx_err)+xx_mean**2/var_x))

prediction_table['Std.Error'] = [std_err_slope, std_err_intercept]

prediction_table['t-Statistic'] = [pred_slope/std_err_slope, pred_intercept/std_err_intercept]


p_value_slope = sps.t.sf(abs(pred_slope/std_err_slope), df = len(xx_err)-1)

p_value_intercept = sps.t.sf(abs(pred_intercept/std_err_intercept), df = len(xx_err)-1)

prediction_table['p-value'] = [p_value_slope, p_value_intercept]

# Since the p-values are basically zero, we find that the predicted values are more than likely correct

X = np.linspace(0,50, 200)

Y = pred_slope*X + pred_intercept

func_string = f'y={pred_slope:.2f}x+{pred_intercept:.2f}'

plt.scatter(main_df['EXP'], main_df['SALARY'], s=3, label= func_string)
plt.plot(X, Y, '-r')
plt.title('Linear Regression between Experience-Salary')
plt.legend(loc='upper left')
plt.grid()


# Checking the results with sklearn

