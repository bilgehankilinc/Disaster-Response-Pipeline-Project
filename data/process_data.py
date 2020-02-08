import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # 1. Import libraries and load datasets.
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    # 2. Merge datasets.
    df = pd.concat([messages, categories], axis=1)

    return df


def clean_data(df):
    # 3. Split categories into separate category columns.
    # get categories column from df for cleaning like in the notebook
    categories = df['categories']

    # create a dataframe of the 36 individual category columns
    categories = categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    # since we took first row there are category target values in our list
    # spliting category boolean values will be a better category representation idea
    splited_list = []
    for category in row.tolist():
        category, _ = category.split('-')
        splited_list.append(category)

    # define category column names with cleaned category names
    category_colnames = splited_list

    # rename the columns of `categories`
    categories.columns = category_colnames

    # 4. Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: str(x[-1]))

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # 5. Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # 6. Remove duplicates.
    # check number of duplicates
    num_of_duplicates = df.message.shape[0] - df.message.nunique()
    # drop duplicates
    df.drop_duplicates(subset='message', inplace=True)
    # check number of duplicates
    num_of_duplicates = df.message.shape[0] - df.message.nunique()
    # 6.5. Drop original column.
    # it seems english social and news messages has no original values
    # since they dont needed any translation while database's creation
    # therefore i've decided to drop this column
    df.drop('original', axis=1, inplace=True)

    return df

"""
def f_word_cloud_gen(df):
    # finally lets try to create a word cloud
    # Libraries for tokenization function
    import matplotlib.pyplot as plt
    import re
    import nltk
    from wordcloud import WordCloud

    nltk.download(['wordnet', 'stopwords', 'punkt'])
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer

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

    text = tokenize(''.join(message for message in df['message'].values))

    w_cloud = WordCloud(width=4800, height=3200, max_words=400, background_color = 'white')
    cloud = w_cloud.generate(' '.join(text))

    cloud.to_file('data/word_cloud.png')
"""

def save_data(df, database_filename):
    # 7. Save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('YourTableName', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        # print('Generating wordcloud...')
        # f_word_cloud_gen(df)
        
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
