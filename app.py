from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("salary_predict_model.pkl")


@app.route("/")
def home():
    return (
        "<h1>Salary Prediction API</h1>"
        "<p>BAIS:3300 - Digital Product Development</p>"
        "<p>Lake Mauer</p>"
    )


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            "age",
            "gender",
            "country",
            "highest_deg",
            "coding_exp",
            "title",
            "company_size",
        ]

        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing one or more required fields"}), 400

        features = [
            int(data["age"]),
            int(data["gender"]),
            int(data["country"]),
            int(data["highest_deg"]),
            int(data["coding_exp"]),
            int(data["title"]),
            int(data["company_size"]),
        ]

        prediction = model.predict([features])[0]

        return jsonify({"predicted_salary": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)