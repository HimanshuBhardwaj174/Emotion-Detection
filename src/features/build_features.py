import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import yaml
import os
print(os.getcwd())

def load_params(path):
    max_feature = yaml.safe_load(open(path,'r'))['feature_eng']['max_features']
    return max_feature

def load_data(train_path,test_path) -> pd.DataFrame:
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data,test_data

def fillna_value(value,train_data,test_data):
    train_data.fillna('hello',inplace=True)
    test_data.fillna('hello',inplace=True)

def save_data(path_,train_df,test_df):
    os.makedirs(path_)

    train_df.to_csv(os.path.join(path_,'train_feature.csv'))
    test_df.to_csv(os.path.join(path_,'test_feature.csv'))


def main(): 
    max_feature = load_params('./params.yaml')
    train_data,test_data = load_data('./data/interim/train_process.csv','./data/interim/test_process.csv')

    fillna_value('hello',train_data,test_data)
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    # # Apply Bag of Words (CountVectorizer)
    vectorizer = CountVectorizer(max_features=max_feature)

    # # Fit the vectorizer on the training data and transform it
    X_train_bow = vectorizer.fit_transform(X_train)

    # # Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(X_test)


    train_df = pd.DataFrame(X_train_bow.toarray())

    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_bow.toarray())

    test_df['label'] = y_test

    save_data(os.path.join('data','processed'),train_df,test_df)

if __name__=='__main__':
    main()





