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

(1) Flask APP
(2) Database
(3) Category Data
(4) Message Data
(5) ETL Process 
(6) ML Pipeline Process


├── app
│   └── run.py (1)
├── data
│   ├── DisasterResponse.db (2)
│   ├── disaster_categories.csv (3)
│   ├── disaster_messages.csv (4)
│   └── process_data.py (5)
├── models
│   └── train_classifier.py (6)


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

### Acknowledgments:

- Sourcing for Readme found: https://github.com/matiassingers/awesome-readme
