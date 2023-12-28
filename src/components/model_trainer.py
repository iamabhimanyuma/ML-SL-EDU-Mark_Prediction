import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from dataclasses import dataclass
from src.utils import model_evaluation,save_obj
from sklearn.metrics import r2_score


@dataclass
class model_trainer_config:
    model_trainer_path_config=os.path.join('artifacts','model.pkl')

class model_trainer:
    
    def __init__(self):
        self.model_trainer_path=model_trainer_config()

    def model_trainer_object(self,train_array,test_array):
        try:
            x_train=train_array[:,:-1]
            y_train=train_array[:,-1]
            x_test=test_array[:,:-1]
            y_test=test_array[:,-1]

            model_params={
                'SVM':{
                    'model':SVR(),
                    'params':{
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf', 'poly'],
                        'epsilon': [0.1, 0.2, 0.3]}},
                'Linear Regression':{
                    'model':LinearRegression(),
                    'params':{
                        'fit_intercept': [True, False]}}}

            model_param={
                'Linear Regression':{
                    'model':LinearRegression(),
                    'params':{
                        'fit_intercept': [True, False]}},
                'Desicion Tree':{
                    'model':DecisionTreeRegressor(),
                    'params':{
                        'max_depth': [None, 5, 10],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['auto', 'sqrt', 'log2']
                            }},
                'SVM':{
                    'model':SVR(),
                    'params':{
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf', 'poly'],
                        'epsilon': [0.1, 0.2, 0.3]
                    }},
                'KNN':{
                    'model':KNeighborsRegressor(),
                    'params':{
                        'n_neighbors': [3, 5, 7],
                        'weights': ['uniform', 'distance'],
                        'p': [1, 2]
                    }},
                'Random Forest':{
                    'model':RandomForestRegressor(),
                    'params':{
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['auto', 'sqrt', 'log2']
                    }},
                'Ada Boost':{
                    'model':AdaBoostRegressor(),
                    'params':{
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }},
                'GB':{
                    'model':GradientBoostingRegressor(),
                    'params':{
                        'learning_rate': [0.01, 0.1, 0.2],
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 1.0]
                    }},
                'XGB':{
                    'model':XGBRegressor(),
                    'params':{
                        'learning_rate': [0.01, 0.1, 0.2],
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 1.0]
                    }},
                'CB':{
                    'model':CatBoostRegressor(),
                    'params':{
                        'iterations': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'depth': [3, 5, 7]
                    }} 
                }
        
            report=model_evaluation(x_train,y_train,x_test,y_test,model_params)
            best_model_score=max(report.values())
            best_model_name=max(report, key=report.get)
            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info('Best model selected')
            best_model=model_params[best_model_name]['model']
            save_obj(filepath=self.model_trainer_path.model_trainer_path_config,obj=best_model)

            predicted=best_model.predict(x_test)
            r2= r2_score(y_test,predicted)
            return r2


        except Exception as e:
            raise CustomException(e,sys)
