# risk-rep
How to execute this code:

After download or clone this git repo:
<br>
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
* And also a name of csv file to execute the fraud prediction model on it
  <br>
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
