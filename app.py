from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import warnings
import pickle
from convert import convertion
from feature import FeatureExtraction

warnings.filterwarnings('ignore')

# Load model
with open("newmodel.pkl", "rb") as file:
    gbc = pickle.load(file)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Enable CORS for all routes
@app.route("/api/test-cors", methods=["GET"])
def test_cors():
    return jsonify({"message": "CORS headers are set"})
@app.route("/")
def home():
    return "Hello, World!"

@app.route("/api/data", methods=["GET"])
def get_data():
    return jsonify({"message": "Data returned"})

@app.route('/api/check-url', methods=['POST'])
def predict():
    try:
        print("Received a request")  # Debug line
        data = request.get_json()
        print(f"Request JSON: {data}")
        url = data.get("url")

        if not url:
            print("No URL in request")
            return jsonify({"error": "Missing 'url' in request"}), 400

        obj = FeatureExtraction(url)
        features = obj.getFeaturesList()
        print(f"Features extracted: {features}")
        x = np.array(features).reshape(1, 30)

        y_pred = gbc.predict(x)[0]
        print(f"Prediction: {y_pred}")

        result = convertion(url, int(y_pred))
        print(f"Result: {result}")

        return jsonify(result)

    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
