# api/app.py

from flask import Flask, request, jsonify
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel


app = Flask(__name__)
 
spark = (
    SparkSession.builder
    .appName("TaxiModelsAPI")
    .master("local[*]")  # local mode
    .getOrCreate()
)

 
def load_spark_model(path: str):
    """
    Tries to load a saved Spark model directory as:
    - CrossValidatorModel
    - TrainValidationSplitModel
    - PipelineModel
    """
    for cls in (CrossValidatorModel, TrainValidationSplitModel, PipelineModel):
        try:
            return cls.load(path)
        except Exception:
            continue
    raise ValueError(f"Could not load Spark model at {path}")


MODEL_BASE = "models"

 
model_highlow = load_spark_model(f"{MODEL_BASE}/high_low_fare_prediction")

 
model_tip = load_spark_model(f"{MODEL_BASE}/tip_prediction")
 
model_payment = load_spark_model(f"{MODEL_BASE}/payment_prediction")


 
REQUIRED_FEATURES = [
    # Categorical
    "VendorID",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_day_of_week",

     
    "trip_distance",
    "trip_duration_min",
    "passenger_count",
    "fare_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",

    # Numerical used by payment model
    "extra",
    "mta_tax",
]


def validate_payload(payload: dict):
    """
    Ensures all required fields exist in the request JSON.
    Returns (ok: bool, error_message: str | None)
    """
    missing = [col for col in REQUIRED_FEATURES if col not in payload]
    if missing:
        return False, f"Missing required field(s): {', '.join(missing)}"
    return True, None


def json_to_spark_df(payload: dict):
    """
    Convert JSON dict into Spark DataFrame with a single row.
    """
    pdf = pd.DataFrame([payload])
    return spark.createDataFrame(pdf)


 
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "Taxi Spark ML API is running.",
        "models": ["highlow", "tip", "payment"]
    })


 
@app.route("/predict/highlow", methods=["POST"])
def predict_highlow():
    try:
        payload = request.get_json(force=True)

        ok, err = validate_payload(payload)
        if not ok:
            return jsonify({"error": err}), 400

        sdf = json_to_spark_df(payload)
        preds = model_highlow.transform(sdf)

         
        row = preds.select("prediction").first()
        if row is None:
            return jsonify({"error": "No prediction generated."}), 500

        pred_value = float(row["prediction"])

         
        label = "High Fare" if pred_value == 1.0 else "Low Fare"

        return jsonify({
            "prediction": label,
            "prediction_index": pred_value
        })

    except Exception as e:
        return jsonify({
            "error": f"Internal error in /predict/highlow: {str(e)}"
        }), 500


 
@app.route("/predict/tip", methods=["POST"])
def predict_tip():
    try:
        payload = request.get_json(force=True)

        ok, err = validate_payload(payload)
        if not ok:
            return jsonify({"error": err}), 400

        sdf = json_to_spark_df(payload)
        preds = model_tip.transform(sdf)

        row = preds.select("prediction").first()
        if row is None:
            return jsonify({"error": "No prediction generated."}), 500

        tip_amount = float(row["prediction"])
        return jsonify({"predicted_tip_amount": round(tip_amount, 2)})

    except Exception as e:
        return jsonify({
            "error": f"Internal error in /predict/tip: {str(e)}"
        }), 500


 
@app.route("/predict/payment", methods=["POST"])
def predict_payment():
    try:
        payload = request.get_json(force=True)

        ok, err = validate_payload(payload)
        if not ok:
            return jsonify({"error": err}), 400

        sdf = json_to_spark_df(payload)
        preds = model_payment.transform(sdf)

        row = preds.select("prediction").first()
        if row is None:
            return jsonify({"error": "No prediction generated."}), 500

        pred_idx = int(row["prediction"])

         
        payment_map = {
            0: "Credit Card",
            1: "Cash",
            2: "No Charge",
            3: "Dispute",
            4: "Unknown/Other"
        }

        return jsonify({
            "predicted_payment_type": payment_map.get(pred_idx, f"class_{pred_idx}"),
            "prediction_index": pred_idx
        })

    except Exception as e:
        return jsonify({
            "error": f"Internal error in /predict/payment: {str(e)}"
        }), 500


if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5001, debug=True)
