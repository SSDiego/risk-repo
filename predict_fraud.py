# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:56:11 2024

@author: Diego de Sousa
"""


from logit_model import model
import pandas as pd 
import numpy as np
import warnings

warnings.filterwarnings("ignore")

csv_file = input("Please enter the dataset for prediction: ")

def predict_fraud(model, csv_file):
    data = pd.read_csv(csv_file)
    predictions = []
    
    for index, row in data.iterrows():
    
        transaction_variables = row[['merchant_id', 'user_id', 'transaction_amount', 'device_id']]  
        transaction_array = np.array(transaction_variables).reshape(1, -1)
        fraud_probability = model.predict_proba(transaction_array)[:, 1]
       
        if fraud_probability[0] >= 0.3:
            print(f"Row {index + 1}: Predicted fraud = Yes")
            predictions.append("Yes")
        else:
            print(f"Row {index + 1}: Predicted fraud = No")
            predictions.append("No")
        
    
    data['Predicted_Fraud'] = predictions
    data.to_csv('log_predictions_model.csv', index=False)



predict_fraud(model, csv_file)
