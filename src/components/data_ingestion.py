import os 
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataPathConfig:
    raw_data_path=os.path.join('artifacts','data.csv')
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')

class DataIngestionConfig:
    def __init__(self):
        self.IngestionConfig=DataPathConfig()
    
    def DataIngestionInit(self):
        try:
            logging.info('Data ingestion initialized')
            df=pd.read_csv('notebook\data\Students.csv')
            logging.info('Data read succesfully')
            train,test=train_test_split(df,test_size=0.2,random_state=43)
            os.makedirs(os.path.dirname(self.IngestionConfig.train_data_path),exist_ok=True)
            df.to_csv(self.IngestionConfig.raw_data_path,header=True, index=False)
            logging.info('Raw data created successfully')
            train.to_csv(self.IngestionConfig.train_data_path, header=True, index=False)
            logging.info('Train data created successfully')
            test.to_csv(self.IngestionConfig.test_data_path, header=True, index=False)
            logging.info('Test data created successfully')
            
        except Exception as e:
            raise CustomException(e,sys)


if __name__=='__main__':
    ing_obj=DataIngestionConfig()
    ing_obj.DataIngestionInit()
