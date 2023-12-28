import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_obj(filepath,obj):
    try:
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        with open(filepath,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def model_evaluation(x_train,y_train,x_test,y_test,model_params):
    try:
        
        scores={}
        for model_name,model_pm in model_params.items():
            gs=GridSearchCV(model_pm['model'],model_pm['params'],cv=5,return_train_score=False)
            gs.fit(x_train,y_train)
            model=model_pm['model']
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            scores[model_name]=test_model_score
        return scores
            
    except Exception as e:
        raise CustomException(e,sys)
def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

