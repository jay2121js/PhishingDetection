# Importing required libraries
from flask import Flask, request, jsonify
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

@app.route("/api/data", methods=["GET"])
def get_data():
    return jsonify({"message": "Data returned"})

@app.route('/api/check-url', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get("url")

        if not url:
            return jsonify({"error": "Missing 'url' in request"}), 400

        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)
        y_pred = gbc.predict(x)[0]

        result = convertion(url, int(y_pred))
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
