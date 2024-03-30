# risk-rep

On a windows pc open the terminal on a folder and:

* Create and activate a new enviroment:

`
python -m venv env
`
`
env\Scripts\activate
` 

* Install the necessary packages

`
pip install requirements.txt
`

* You should have a link for a csv file that will be used for testing the model
* And also csv file name to execute the fraud prediction model on it
  for the second data you can write: train_data.csv

`
https://linkparacsv.csv
`
`
data-for-fraud-classification.csv
`

* On the terminal execute:

`
python predict_fraud.py
`

You will see each transaction being classified one by one. and a log file will be generated in the folder with the results. log_predictions_model.csv
