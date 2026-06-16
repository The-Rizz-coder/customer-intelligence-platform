import os
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.utils import save_object


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = (
            ModelTrainerConfig()
        )

    def initiate_model_training(
        self,
        train_arr,
        test_arr
    ):

        print("Model Training Started...")

        X_train = train_arr[:, :-1]
        y_train = train_arr[:, -1]

        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        models = {

            "Logistic Regression":
                LogisticRegression(),

            "Random Forest":
                RandomForestClassifier(),

            "Gradient Boosting":
                GradientBoostingClassifier()
        }

        best_model = None
        best_model_name = None
        best_f1_score = 0

        for model_name, model in models.items():

            print(f"\nTraining {model_name}...")

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

            if f1 > best_f1_score:

                best_f1_score = f1
                best_model = model
                best_model_name = model_name

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )

        print("\nBest Model:", best_model_name)
        print("Best F1 Score:", best_f1_score)

        print("\nModel Saved Successfully.")

        return best_f1_score
if __name__ == "__main__":

    from src.components.data_ingestion import (
        DataIngestion
    )

    from src.components.data_transformation import (
        DataTransformation
    )

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    transformation = DataTransformation()

    train_arr, test_arr, _ = (
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    trainer = ModelTrainer()

    trainer.initiate_model_training(
        train_arr,
        test_arr
    )