import os
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pickle


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, model):
    model_report = {}
    for model_name, model_instance in model.items():
        try:
            logging.info(f'Training {model_name}')
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            model_report[model_name] = r2_score(y_test, y_pred)

            
            logging.info(f'{model_name} trained successfully')

            
        except Exception as e:
            logging.error(f'Error occurred while training {model_name}')
            logging.error(e)
            raise CustomException(f'Error occurred while training {model_name}')
    return model_report