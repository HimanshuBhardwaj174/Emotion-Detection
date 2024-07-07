import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

import logging
logger = logging.getLogger('data_ingestion')
handler = logging.StreamHandler()
fhandler = logging.FileHandler('error.log')
formatter =  logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -  %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(fhandler)

logger.setLevel('DEBUG')
handler.setLevel('DEBUG') 
fhandler.setLevel('DEBUG')



logger.info('IN THE DATA INGESTION')


def load_data(url:str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

def param_load(path:str) -> float:
    test_size = yaml.safe_load(open(path,'r'))['data_ingestion']['test_size']
    return test_size

def operation(df:pd.DataFrame)->pd.DataFrame:
    df.drop(columns=['tweet_id'],inplace=True)
    final_df = df[df['sentiment'].isin(['happiness','sadness'])]
    final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
    return final_df

def save(path_,train_data,test_data) -> None:
    os.makedirs(path_)

    train_data.to_csv(os.path.join(path_,'train.csv'))
    test_data.to_csv(os.path.join(path_,'test.csv'))

def main():

    df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    
    test_size = param_load('params.yaml')
    logger.info(f'Test size -> {test_size}')

    final_df = operation(df)

    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    save(os.path.join('data','raw'),train_data,test_data)


if __name__=='__main__':
    main()
    
