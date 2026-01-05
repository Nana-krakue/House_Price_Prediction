from flask import Flask, request, jsonify
import pickle, pandas as pd

app = Flask(__name__)

with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sample_df = pd.DataFrame([data])
    pred_price = model.predict(sample_df)[0]
    return jsonify({"predicted_price": round(pred_price, 2)})

if __name__ == "__main__":
    app.run(debug=True)
