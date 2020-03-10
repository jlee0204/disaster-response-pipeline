# Disaster Response Pipeline Project

## Dependencies
<pre>
nltk
json
numpy
pandas
scikit-learn
sqlalchemy
plotly
flask
re
pickly
sys

</pre>
## Files

<pre>

├── app
│   └── run.py------------------------# FLASK FILE THAT RUNS APP
├── data
│   ├── DisasterResponse.db-----------# DATABASE TO SAVE CLEANED DATA TO
│   ├── disaster_categories.csv-------# CATEGOrY DATA
│   ├── disaster_messages.csv---------# MESSAGE DATA
│   └── process_data.py---------------# PERFORMS ETL PROCESS
├── models
│   └── train_classifier.py-----------# PERFORM ML PIPELINE PROCESS

</pre>



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
