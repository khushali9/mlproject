import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns =[
                "gender",
                "race_ethnicity", 
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline=Pipeline(
                steps=[
                    #used for missing value imputation
                    ("imputer", SimpleImputer(strategy="median")),
                    #used for scaling the data
                    ("scaler", StandardScaler()) 
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    #used for missing value imputation
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    #used for encoding categorical data
                    ("one_hot_encoder", OneHotEncoder()),
                    #scaling the data
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and categorical pipelines created successfully.")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_columns),
                    ("cat_pipeline",cat_pipeline, categorical_columns)
                ]
            )
            logging.info("Column transformer created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error occurred while getting data transformer object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self,traing_path,test_path):
        try:
            train_df=pd.read_csv(traing_path)
            test_df=pd.read_csv(test_path)
            logging.info("Data read successfully from the CSV files.")
            logging.info("Obtaining preprocessing object.")
            preprocessing_obj=self.get_data_transformer_object()
            logging.info("Preprocessing object obtained successfully.")
            target_column_name="math_score"
            numerical_columns=["writing_score", "reading_score"]
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Train and test features prepared successfully.")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info("Train and test features transformed successfully.")
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Training data array created successfully.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error occurred during data transformation: {e}")
            raise CustomException(e, sys)
