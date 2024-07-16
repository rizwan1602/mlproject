import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.utils import logging
from src.utils import save_object , evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Initiating Model Training')
            X_train , y_train ,X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'KNN': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostClassifier(),
                'AdaBoost': AdaBoostRegressor(),

            }

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test = X_test ,y_test = y_test,model=model)

            ## sort best model score from dict 

            best_model_score = max(sorted(model_report.values()))

            ##get best model

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model[best_model_name]

            if best_model_score < 0.5:
                logging.error('Model score is less than 0.5')
                raise CustomException('Model score is less than 0.5')
            
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            predicted = best_model.predict(X_test)

            r2_score_data = r2_score(y_test,predicted)

            return r2_score_data


        except Exception as e:
           raise CustomException(e, sys)
