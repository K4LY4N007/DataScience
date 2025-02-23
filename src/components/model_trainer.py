import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
    RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, load_object

@dataclass
class ModelTrainerConfig:
    model_obj_file_path: str=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        """
        This method is used to train the model
        :param train_data: training data
        """
        logging.info('Entered initiate_model_training method')
        try:
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False),
                'LightGBM': LGBMRegressor()
            }

            model_report:dict= evaluate_models(
                X_train= X_train,
                y_train=y_train,
                X_test=X_test, 
                y_test=y_test, 
                models=models
                )
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best Model found')

            save_object(self.model_trainer_config.model_obj_file_path, best_model)

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            # preprocessing_obj =
            # for model_name, model in models.items():
            #     logging.info(f'Training {model_name} model')
            #     model.fit(X_train, y_train)
            #     logging.info(f'{model_name} model trained successfully')

            #     save_object(os.path.join('artifacts', f'{model_name}.pkl'), model)
            #     logging.info(f'{model_name} model saved successfully')

        except Exception as e:
            raise CustomException(e, sys)