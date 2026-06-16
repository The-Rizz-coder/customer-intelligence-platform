import pandas as pd

from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(
        self,
        features
    ):

        preprocessor = load_object(
            "artifacts/preprocessor.pkl"
        )

        model = load_object(
            "artifacts/model.pkl"
        )

        data_scaled = preprocessor.transform(
            features
        )

        prediction = model.predict(
            data_scaled
        )

        return prediction


class CustomData:

    def __init__(
        self,
        Age,
        Gender,
        Tenure,
        MonthlyCharges,
        Contract,
        PaymentMethod,
        TotalCharges
    ):

        self.Age = Age
        self.Gender = Gender
        self.Tenure = Tenure
        self.MonthlyCharges = MonthlyCharges
        self.Contract = Contract
        self.PaymentMethod = PaymentMethod
        self.TotalCharges = TotalCharges

    def get_data_as_dataframe(
        self
    ):

        data_dict = {

            "Age": [self.Age],

            "Gender": [self.Gender],

            "Tenure": [self.Tenure],

            "MonthlyCharges": [
                self.MonthlyCharges
            ],

            "Contract": [
                self.Contract
            ],

            "PaymentMethod": [
                self.PaymentMethod
            ],

            "TotalCharges": [
                self.TotalCharges
            ]
        }

        return pd.DataFrame(
            data_dict
        )