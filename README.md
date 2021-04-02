# Disaster Response Pipeline Project 
A web appp that classify a text message to multiple categories.

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Demo](#demo)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1. Create an anaconda environment and then install the requirements.

    ```
    conda create --name myenv python=3.8
    conda activate myenv 
    pip install -r requirements.txt
    ```


2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/


## Project Motivation<a name="motivation"></a>

The main goal of this project is to build a end-to-end machine learning application following the steps below:

1. ETL Pipeline that load and clean data raw data
2. Machine learning pipeline that train, evaluate and persist a model
3. Web application that loads the trained model, receive a text message input from the user and classify the message to  36 different categories

**Technical details*:
The project is built in Python and use some libraries like pandas, scikit-learn, nltk, flask, plotly, sqlalchemy.
For text message encoding, CountVectorizer and TF-IDF are used.

## File Descriptions <a name="files"></a>

**notebook** folder contains 2 notebooks which are used to prototype the 
etl pipeline and machine learning pipeline. 

**data** folder contains python scripts to process the etl pipeline 
and sample raw data

**models** folder contains python scripts to process the 
machine learning pipeline.

**app**  folder contains the flask web application that process the text messages
using a machine learning model.


## Demo <a name="demo"></a>

A demo of this web application hosted on Heroku is available [here](https://flask-plotly-accommodation.herokuapp.com/).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity for the web application template that is provided
on Udacity's Data Science Nanodegree.

Otherwise, feel free to use the code here as you would like! 