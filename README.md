# Disaster Response Pipeline Project
A project in which we analyse the messages that are recieved during the time of a disaster and categorize them into different categories such as hospitals,first-aid,infrastructure etc thereby segregating a message into a number of categories.Here,we are using data from two source files:messages.csv and categories.csv.We clean the data from these two source file using process_data.py and store the cleaned dataframe in a database named 'Disaster.db'.The 'train_classifier.py' file creates a machine learning pipeline which uses this data and seggregate the messages into different categories and stores the model in a pickle file.Using the model we display the catgories of the message interactively through a flask app. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important Files:
- data/process_data.py: The ETL pipeline used to process data in preparation for model building.
- models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle (pickle is not uploaded to the repo due to size constraints.).
- app/templates/*.html: HTML templates for the web app.
- run.py:This file is used to run the web app. Start the Python server for the web app and prepare visualizations.
