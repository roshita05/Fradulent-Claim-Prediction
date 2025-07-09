from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

# Mappings for categorical inputs
sex_map = {"MALE": 0, "FEMALE": 1}
edu_map = {"High School": 0, "Associate": 1, "College": 2, "Masters": 3, "Doctorate": 4}
hobby_map = {"Reading": 0, "Sports": 1, "Music": 2, "Gaming": 3, "Art": 4, "Travel": 5}
rel_map = {"Husband": 0, "Wife": 1, "Other": 2, "Not-in-family": 3}
price_map = {"Low": 0, "Medium": 1, "High": 2}
policy_map = {"Liability": 0, "Collision": 1, "All Perils": 2}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Fetch form data
        insured_sex = request.form.get("insured_sex")
        insured_education = request.form.get("insured_education")
        insured_hobbies = request.form.get("insured_hobbies")
        insured_relationship = request.form.get("insured_relationship")
        vehicle_price = request.form.get("vehicle_price")
        base_policy = request.form.get("base_policy")

        policy_deductable = int(request.form.get("policy_deductable"))
        days_policy_inception = int(request.form.get("days_policy_inception"))
        umbrella_limit = int(request.form.get("umbrella_limit"))
        capital_gains = int(request.form.get("capital_gains"))
        capital_loss = int(request.form.get("capital_loss"))
        incident_hour_of_the_day = int(request.form.get("incident_hour_of_the_day"))

        # Convert to dataframe
        input_df = pd.DataFrame([{
            'insured_sex': sex_map.get(insured_sex),
            'insured_education': edu_map.get(insured_education),
            'insured_hobbies': hobby_map.get(insured_hobbies),
            'insured_relationship': rel_map.get(insured_relationship),
            'vehicle_price': price_map.get(vehicle_price),
            'base_policy': policy_map.get(base_policy),
            'policy_deductable': policy_deductable,
            'days_policy_inception': days_policy_inception,
            'umbrella_limit': umbrella_limit,
            'capital-gains': capital_gains,
            'capital-loss': capital_loss,
            'incident_hour_of_the_day': incident_hour_of_the_day
        }])

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "🚨 Fraud Detected!" if prediction == 1 else "✅ Legitimate Claim"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"<h2 style='color:red;'>Error: {str(e)}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
