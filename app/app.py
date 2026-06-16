import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
) 
from flask import (
    Flask,
    render_template,
    request
)

from src.pipeline.predict_pipeline import (
    PredictPipeline,
    CustomData
)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = CustomData(

        Age=int(
            request.form.get("Age")
        ),

        Gender=request.form.get(
            "Gender"
        ),

        Tenure=int(
            request.form.get("Tenure")
        ),

        MonthlyCharges=float(
            request.form.get(
                "MonthlyCharges"
            )
        ),

        Contract=request.form.get(
            "Contract"
        ),

        PaymentMethod=request.form.get(
            "PaymentMethod"
        ),

        TotalCharges=float(
            request.form.get(
                "TotalCharges"
            )
        )
    )

    pred_df = (
        data.get_data_as_dataframe()
    )

    predict_pipeline = (
        PredictPipeline()
    )

    prediction = (
        predict_pipeline.predict(
            pred_df
        )
    )

    result = (
        "Customer Likely To Churn"
        if prediction[0] == 1
        else
        "Customer Likely To Stay"
    )

    return render_template(
    "home.html",
    prediction_text=result,
    prediction_value=prediction[0]
)


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )