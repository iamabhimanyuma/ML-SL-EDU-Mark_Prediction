import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformPathConfig:
    data_transform_path_config=os.path.join('artifacts','preprocessor.pkl')

class DataTransformConfig:
    def __init__(self):
        self.data_transform_path=DataTransformPathConfig()

    def DataPreprocessingObject(self):
        try:
            cat_var=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_var=['reading_score','writing_score']

            num_pipeling=Pipeline(steps=[
                ('simple_imputer',SimpleImputer(strategy='median')),
                ('StdScaler',StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ('simple_imputer',SimpleImputer(strategy='most_frequent')),
                ('OHE',OneHotEncoder()),
                ('StdScaler',StandardScaler(with_mean=False))
                ])
            preprocessor=ColumnTransformer([
                ('num_preprocessing',num_pipeling,num_var),
                ('cat_preprocessing',cat_pipeline,cat_var)
            ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def DataTransformInitiation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            x_train=train_data.drop('math_score',axis=1)
            y_train=train_data['math_score']
            x_test=test_data.drop('math_score',axis=1)
            y_test=test_data['math_score']
            

            preprocessing_obj=self.DataPreprocessingObject()
            x_train=preprocessing_obj.fit_transform(x_train)
            x_test=preprocessing_obj.transform(x_test)
            train_arr=np.c_[x_train,np.array(y_train)]
            test_arr=np.c_[x_test,np.array(y_test)]
            
            logging.info('Saved preprocessing object')
            save_obj(filepath=self.data_transform_path.data_transform_path_config,obj=preprocessing_obj)
        
            return (
                train_arr,
                test_arr,
                self.data_transform_path.data_transform_path_config
        )

    

            
        except Exception as e:
            raise CustomException(e,sys) 

    
