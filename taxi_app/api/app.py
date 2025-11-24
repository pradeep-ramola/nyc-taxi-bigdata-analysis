from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from utils import load_model, preprocess_input

app = Flask(__name__)

# Load three PyTorch models
model_highlow = load_model("models/high_low_fare_prediction.pt")
model_tip = load_model("models/tip_prediction.pt")
model_payment = load_model("models/payment_prediction.pt")

@app.route("/")
def index():
    return "Taxi ML API is running."

# ------- HIGH VS LOW FARE -------
@app.route("/predict/highlow", methods=["POST"])
def predict_highlow():
    data = request.json
    x = preprocess_input(data)
    with torch.no_grad():
        pred = model_highlow(x)
        label = "High Fare" if pred.argmax() == 1 else "Low Fare"
    return jsonify({"prediction": label})

# ------- TIP PREDICTION -------
@app.route("/predict/tip", methods=["POST"])
def predict_tip():
    data = request.json
    x = preprocess_input(data)
    with torch.no_grad():
        pred = model_tip(x).item()
    return jsonify({"predicted_tip_amount": round(pred, 2)})

# ------- PAYMENT TYPE PREDICTION -------
@app.route("/predict/payment", methods=["POST"])
def predict_payment():
    data = request.json
    x = preprocess_input(data)
    with torch.no_grad():
        pred = model_payment(x).argmax().item()

    payment_map = {0: "Cash", 1: "Credit Card", 2: "No Charge", 3: "Dispute"}
    return jsonify({"predicted_payment_type": payment_map.get(pred, "Unknown")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
