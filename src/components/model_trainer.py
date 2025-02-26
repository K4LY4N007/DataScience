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
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'XGBRegressor': XGBRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(verbose=False)
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [4,6,8,10,16,32,64]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [4,6,8,10,16,32],
                    'max_depth': [3, 5, 7]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001,0.000125],
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth': [3, 5, 7]
                },
                "CatBoosting Regressor":{
                    'depth': [2,4,6,8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001,0.000125],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [4,6,8,10,16,32,64]
                }
                
            }

            model_report:dict= evaluate_models(
                X_train= X_train,
                y_train=y_train,
                X_test=X_test, 
                y_test=y_test, 
                models=models,
                param=params
                )
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            print(best_model_name, best_model)
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