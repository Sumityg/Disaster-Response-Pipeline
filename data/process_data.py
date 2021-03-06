import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Reads the message and the categories file and returns merged dataframe
    containing both messages and their categories"""
    messages = pd.read_csv(messages_filepath)
    categories =pd.read_csv(categories_filepath)
    df = messages.merge(categories,left_on='id',right_on='id')
    return df


def clean_data(df):
    """Cleaning the dataframe and returning a cleaned dataframe """
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x :x[:-2])
    for column in categories:
        categories[column] = categories[column].astype(str)
        categories[column] =categories[column].apply(lambda x :x[-1])
        categories[column] = pd.to_numeric(categories[column])
    categories.rename(category_colnames,axis=1,inplace=True)
    df=df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df['related']=df['related'].replace(2,1)
    df=df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """Saving the clean dataframe to the database"""
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Message', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()