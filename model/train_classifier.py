import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk import word_tokenize
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer  
nltk.download('wordnet')
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pickle


def load_data(database_filepath):
    """ 
    A function that reads the data from the table and returns the input variable,target variables and the target categories
    as output
    Extended description of function. 
  
    Parameters: 
    database_filepath: The database in which the table is stored
  
    Returns: 
    input variable,target variables and the target categories
    as output
  
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Message',con=engine)
    X =df.message 
    Y =df.drop(columns=['message','id','original','genre'],axis=1) 
    category_names=Y.columns
    return X,Y,category_names


def tokenize(text):
    """ 
    A function that takes text as an input,normalises and tokenise it to return the tokens 
    as output
    Extended description of function. 
  
    Parameters: 
    text: text data that we have to tokenize 
  
    Returns: 
    tokens from the input text
  
    """
    text=text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text=word_tokenize(text)
    stop_words=stopwords.words('english')
    words = [w for w in text if w not in stop_words]
    words=[]
    for w in text: 
        if w not in stop_words:
            words.append(w) 
    clean_tokens=[]
    lemmatizer = WordNetLemmatizer()
    for w in words:
        clean_tok=lemmatizer.lemmatize(w)
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """ 
    A function that builds a model and applies Grid search on its Parameters
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'vect__min_df': [5],
                  'tfidf__use_idf':[True],
                  'clf__estimator__n_estimators':[10], 
                  'clf__estimator__min_samples_split':[10]}

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 10,n_jobs=1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    A function that takes the model,testing dataset and the category_names as input and returns 
    the classification report which contains the f-score,precision and recall score
    as output
    Extended description of function. 
  
    Parameters: 
    model: the classifier used for classification
    X_test,Y_test:the testing dataset
    category_names:the categories that would be assigned to each of the input message
  
    Returns: 
    classification report which contains the f-score,precision and recall score
  
    """
    Y_pred=model.predict(X_test)
    print(classification_report(Y_pred,Y_test.values,target_names=category_names))
    

def save_model(model, model_filepath):
    """ 
    A function that takes the model as saves it to the disk
    Extended description of function. 
  
    Parameters: 
    model: the classifier used for classification
    model_filepath:the filepath where we have to save the model
    
    """
    
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()