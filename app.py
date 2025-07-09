
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("fraud_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sex = request.form.get("insured_sex")
    education = request.form.get("insured_education")
    hobbies = request.form.get("insured_hobbies")
    relationship = request.form.get("insured_relationship")
    vehicle_price = request.form.get("vehicle_price")
    base_policy = request.form.get("base_policy")
    policy_deductable = int(request.form.get("policy_deductable"))
    days_policy_inception = int(request.form.get("days_policy_inception"))
    umbrella_limit = int(request.form.get("umbrella_limit"))
    capital_gains = int(request.form.get("capital_gains"))
    capital_loss = int(request.form.get("capital_loss"))
    incident_hour_of_the_day = int(request.form.get("incident_hour_of_the_day"))

    # Encode features
    sex_map = {"MALE": 0, "FEMALE": 1}
    edu_map = {"High School": 0, "Associate": 1, "College": 2, "Masters": 3, "Doctorate": 4}
    hobby_map = {"Reading": 0, "Sports": 1, "Music": 2, "Gaming": 3, "Art": 4, "Travel": 5}
    rel_map = {"Husband": 0, "Wife": 1, "Other": 2, "Not-in-family": 3}
    price_map = {"Low": 0, "Medium": 1, "High": 2}
    policy_map = {"Liability": 0, "Collision": 1, "All Perils": 2}

    input_df = pd.DataFrame([{
        'insured_sex': sex_map[sex],
        'insured_education': edu_map[education],
        'insured_hobbies': hobby_map[hobbies],
        'insured_relationship': rel_map[relationship],
        'vehicle_price': price_map[vehicle_price],
        'base_policy': policy_map[base_policy],
        'policy_deductable': policy_deductable,
        'days_policy_inception': days_policy_inception,
        'umbrella_limit': umbrella_limit,
        'capital-gains': capital_gains,
        'capital-loss': capital_loss,
        'incident_hour_of_the_day': incident_hour_of_the_day
    }])

    prediction = model.predict(input_df)[0]
    result = "🚨 Fraud Detected!" if prediction == 1 else "✅ Legitimate Claim"
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
