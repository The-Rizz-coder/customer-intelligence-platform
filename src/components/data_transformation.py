import os
import sys

import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from src.utils import save_object


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = (
            DataTransformationConfig()
        )

    def get_data_transformer_object(self):

        numerical_columns = [
            "Age",
            "Tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        categorical_columns = [
            "Gender",
            "Contract",
            "PaymentMethod"
        ]

        num_pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num_pipeline",
                    num_pipeline,
                    numerical_columns
                ),

                (
                    "cat_pipeline",
                    cat_pipeline,
                    categorical_columns
                )
            ]
        )

        return preprocessor

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        print("Data Transformation Started...")

        train_df = pd.read_csv(train_path)

        test_df = pd.read_csv(test_path)

        train_df = train_df.drop(
            columns=["CustomerID"]
        )

        test_df = test_df.drop(
            columns=["CustomerID"]
        )

        target_column = "Churn"

        X_train = train_df.drop(
            columns=[target_column]
        )

        y_train = train_df[target_column]

        X_test = test_df.drop(
            columns=[target_column]
        )

        y_test = test_df[target_column]

        y_train = y_train.map({
            "No": 0,
            "Yes": 1
        })

        y_test = y_test.map({
            "No": 0,
            "Yes": 1
        })

        preprocessor = (
            self.get_data_transformer_object()
        )

        X_train_transformed = (
            preprocessor.fit_transform(X_train)
        )

        X_test_transformed = (
            preprocessor.transform(X_test)
        )

        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor
        )

        train_arr = np.c_[
            X_train_transformed,
            y_train
        ]

        test_arr = np.c_[
            X_test_transformed,
            y_test
        ]

        print("Data Transformation Completed.")

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )
if __name__ == "__main__":

    from src.components.data_ingestion import (
        DataIngestion
    )

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    print("\nTrain Array Shape:", train_arr.shape)
    print("Test Array Shape :", test_arr.shape)

    print("\nPreprocessor Saved At:")
    print(preprocessor_path)