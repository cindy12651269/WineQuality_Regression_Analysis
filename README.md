# WineQuality_Regression_Analysis
This Python script demonstrates how to perform a regression analysis on the winequality-both dataset to predict wine quality based on various features. The dataset is divided into training and testing sets, and the Ordinary Least Squares (OLS) method is used for the regression analysis.

## Sample Code

```python
import pandas as pd
import statsmodels.api as sm

source = 'https://raw.githubusercontent.com/cbrownley/foundations-for-analytics-with-python/master/statistics/winequality-both.csv'
df = pd.read_csv(source)

# Use the last 10 rows as the test data
test = df.tail(10)
testy = test['quality']
testX = sm.add_constant(test[test.columns.difference(['type', 'quality'])])

# Use the remaining data as the training data
train = df.iloc[:-10]
y = train['quality']
X = sm.add_constant(train[train.columns.difference(['type', 'quality'])])

# A: Train the regression model
model = sm.OLS(y, X).fit()

# B: Output the model summary
print(model.summary())

# C: Use the trained model to predict the test data
print(model.predict(testX))
