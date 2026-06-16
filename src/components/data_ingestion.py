import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "raw.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        print("Data Ingestion Started...")

        dataset_path = os.path.join(
            "data",
            "raw",
            "customer_churn.csv"
        )

        df = pd.read_csv(dataset_path)

        # Create artifacts directory
        os.makedirs(
            os.path.dirname(
                self.ingestion_config.train_data_path
            ),
            exist_ok=True
        )

        # Save raw dataset
        df.to_csv(
            self.ingestion_config.raw_data_path,
            index=False
        )

        print("Raw data saved successfully.")

        # Train-Test Split
        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["Churn"]
        )

        # Save train dataset
        train_set.to_csv(
            self.ingestion_config.train_data_path,
            index=False
        )

        # Save test dataset
        test_set.to_csv(
            self.ingestion_config.test_data_path,
            index=False
        )

        print("Train and Test files saved successfully.")

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )


if __name__ == "__main__":

    obj = DataIngestion()

    train_path, test_path = obj.initiate_data_ingestion()

    print("\nTrain Path:", train_path)
    print("Test Path :", test_path)



