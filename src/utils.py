import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import dill

def save_object(file_path, obj):
    """
    This method is used to save the object in the file
    :param obj: object to be saved
    :param file_path: file path where object is to be saved
    """
    try:
        logging.info('Entered save_object method')
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

        logging.info('Object saved successfully')
    except Exception as e:
        raise CustomException(e, sys)