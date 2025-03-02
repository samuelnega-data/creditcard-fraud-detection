# Credit Card Fraud Detection

## Overview 
In this project, I will analyze 28 different variables to determine which are most effective in detecting credit card fraud. The analysis will be conducted using Python in Jupyter Notebook.

## Code 
## Importing Libraries
Imports essential libraries for data manipulation (pandas, numpy), visualization (seaborn, matplotlib, plotly), and machine learning (scikit-learn).
```pyt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
## Importing Libraries
Reads the dataset from a CSV file into a Pandas DataFrame.
```pyt
df = pd.read_csv(r"C:\Users\samue\OneDrive\Desktop\creditcard.csv")
```
## Basic dataset information
Viewing the datset structure, datatypes and looking for any missing values. I also viewed a summary of the statistics of each column. I also wanted to further explore the standard deviation to have a better understanding of outliers in the dataset.
```pyt
df.info()
df.describe()
df.describe().loc["std", df.columns[1:]]
```

## Count of fraud vs non-fraud transactions
```pyt
print(df['Class'].value_counts())
```

## Visualizing fraudulent vs non-fraudulent transactions over time
Extracts transaction times for both fraud (Class = 1) and non-fraud (Class = 0) transactions and created overlapping distribution plots of transaction times for both groups from class.
```pyt
class0 = df.loc[df['Class'] == 0]["Time"]
class1 = df.loc[df['Class'] == 1]["Time"]

hist_data = [class0, class1]
group_labels = ['Not Fraudulent', 'Fraudulent']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig.update_layout(title='Credit Card Transactions Vs Time', xaxis=dict(title='Time'))
fig.show()
```

## Distribution of transaction amounts for fraud vs non-fraud cases
Created histograms comparing transaction amounts for fraudulent and non-fraudulent transactions.
```pyt
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

bins = 30

ax1.hist(df.Amount[df.Class == 1], bins=bins, color='red', alpha=0.7)
ax1.set_title('Fraudulent Transactions')

ax2.hist(df.Amount[df.Class == 0], bins=bins, color='blue', alpha=0.7)
ax2.set_title('Non-Fraudulent Transactions')

plt.xlabel('Transaction Amount')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()
```
## Correlation heatmap
Computed correlations between features and visualizes them using a heatmap with seaborn.
```pyt
plt.figure(figsize=(12, 10))
plt.title('Credit Card Transactions Correlation Map')
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, linewidths=2)
plt.show()
```
# Train Split Test
The correlation map showed no strong correlations between the features, meaning no single variable could reliably predict fraud on its own. Given this, a train-test split was a better approach to assess model performance, ensuring the model learns patterns from diverse data rather than relying on a single misleading feature.

## Defining Features and Target Variable
```pyt
X = df.drop(columns='Class', axis=1)
Y = df['Class']
```
## Splitting data (80/20 Split)
```pyt
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```
## Display Dataset Shapes
```pyt
print("Dataset Shape:", X.shape)
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
```

## Logistic Regression Model & Training the Model
Created a Logistic Regression model and sets max_iter=1000 to avoid convergence warnings.
```pyt
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)
```
## Model evaluation
Used the trained model to predict the class labels for X_train.
```pyt
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on Training Data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on Test Data:', test_data_accuracy)
```




