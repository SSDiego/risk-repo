# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:15:42 2024

@author: Diego
"""

# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from utils import select_columns, scatter_plot, logistic_curve
import matplotlib.pyplot as plt
import numpy as np



# Data processing
url = input("Please enter the dataset URL: ")
df = pd.read_csv(url)
df_work = select_columns(df, [1, 2, 5, 6, 7], 'has_cbk')
#scatter_plot(df_work, 'transaction_amount', 'has_cbk', 'transactions')
target = df_work['has_cbk'].to_list()
df_work.drop(columns=['has_cbk'], inplace=True)


# Model Fitting
X_train, X_test, y_train, y_test = train_test_split(df_work, target, test_size=0.7, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:,1]
score_table = model.predict_proba(X_test)


#Analysing the Odds
predicted_probabilities = model.predict_proba(X_test)
predicted_probabilities_positive = predicted_probabilities[:, 1]
odds = predicted_probabilities_positive / (1 - predicted_probabilities_positive)
log_odds = np.log(odds)
print(predicted_probabilities)
print(predicted_probabilities_positive)
print(odds)
print(log_odds)

odds_df = pd.DataFrame({
    'Predicted_Probability': predicted_probabilities_positive,
    'Odds': odds,
    'Log_Odds': log_odds
})
odds_df['y_test'] = y_test
compare = np.exp(log_odds)/(1+ np.exp(log_odds))
odds_df['compared'] = compare
odds_df['class_0_probability'] = score_table[:, 0]
odds_df['class_1_probability'] = score_table[:, 1]

odds_df.to_csv('odds_data.csv')

#Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = model.predict(X_test)


#Goodness
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)
conf_matrix = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'AUC Score: {auc}')
print('Confusion Matrix:')
print(conf_matrix)


# alternative_threshold = 0.6
# predicted_outcomes_alt_threshold = (cc_lr.predict_proba(X_test)[:,1] >= alternative_threshold).astype(int)


train_probabilities = model.predict_proba(X_train)[:, 1]
train_probab_range = model.predict_proba(X_train)
train_predictions = model.predict(X_train)

X_test['probab'] = probabilities
X_test['predict'] = predictions

X_train['probab'] = train_probabilities = model.predict_proba(X_train)[:, 1]
X_train['predict'] = train_predictions
X_train['y_train'] = y_train

X_train.to_csv('train_data.csv')

