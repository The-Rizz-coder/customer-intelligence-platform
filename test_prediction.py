from src.pipeline.predict_pipeline import (
    PredictPipeline,
    CustomData
)

sample = CustomData(

    Age=45,

    Gender="Male",

    Tenure=36,

    MonthlyCharges=80,

    Contract="One year",

    PaymentMethod="Credit card",

    TotalCharges=2880
)

df = sample.get_data_as_dataframe()

predict_pipeline = PredictPipeline()

prediction = predict_pipeline.predict(
    df
)

print(prediction)