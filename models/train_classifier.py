import sys
# Libraries for loading
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
# Libraries for tokenization function
import re
import nltk

nltk.download(['wordnet', 'stopwords', 'punkt'])
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Libraries for ml pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

# Random State
np.random.seed(42)


def load_data(database_filepath):
    # 1. Import libraries and load data from database. data/DisasterResponse.db
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM YourTableName', engine)

    X = df['message']
    Y = df.drop(['message', 'genre'], axis=1)
    column_names = Y.columns.tolist()
    return X, Y, column_names


def tokenize(text):
    # 2. Write a tokenization function to process your text data
    '''
    This function takes a text
    cleans from punctiation, tokenize it
    and remove english stop words
    then returns list constructed by tokenized text elements.

    input : a collection of test messages such as list/dataframe.
    output : a cleaned and tokenized list.
    '''

    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = word_tokenize(text)
    wnlm = WordNetLemmatizer()
    words = [wnlm.lemmatize(word) for word in words if word not in stopwords.words('english')]

    return words


def build_model():
    # 3. Build a machine learning pipeline
    final_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize, max_features=5000)),
        ('transformer', TfidfTransformer(use_idf=True)),
        ('classifer', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
    ])

    return final_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # Making prediction from our test data (25% of our original data)
    y_pred = pd.DataFrame(model.predict(X_test), index=Y_test.index, columns=category_names)

    for column in category_names:
        cr_default = classification_report(Y_test[column].values, y_pred[column].values)
        print('Column Name: ', column)
        print('Classification Report: ', cr_default)


def save_model(model, model_filepath):
    # Saving final model.
    f = model_filepath
    with open(f, 'wb') as file:
        pickle.dump(model, file)



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