from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('app/model/best_model.pkl')

# Feature names (auto-extracted)
features = ['Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Age', 'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice', 'PastNumberOfClaims', 'AgeOfVehicle', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfCars', 'BasePolicy']

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [request.form.get(f) for f in features]
        input_df = pd.DataFrame([input_data], columns=features)
        input_df = input_df.apply(pd.to_numeric, errors='ignore')

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = {
            'prediction': 'Fraudulent' if prediction == 1 else 'Not Fraudulent',
            'probability': f"{probability:.2%}"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)