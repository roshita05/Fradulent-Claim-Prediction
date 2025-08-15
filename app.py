import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load trained model (pipeline with preprocessing inside)
# ---------------------------
model = joblib.load('models/fraud_rf.pkl')

# ---------------------------
# Function to load predefined fraud test cases
# ---------------------------
def load_test_case(case_name):
    test_cases = {
        "High Fraud Risk 1": {
            'Sex': "Male",
            'Age': 25,
            'VehiclePrice': 85000,
            'VehicleCategory': "SUV",
            'MaritalStatus': "Single",
            'Fault': "Driver",
            'PolicyType': "Premium",
            'AgeOfVehicle': 1,
            'AgeOfPolicyHolder': 25,
            'Days_Policy_Accident': 5,
            'Days_Policy_Claim': 2,
            'PoliceReportFiled': "No",
            'WitnessPresent': "No",
            'NumberOfSuppliments': 3,
            'AddressChange_Claim': "Yes",
            'NumberOfCars': 1,
            'Deductible': 500,
            'BasePolicy': "All Perils",
            'PastNumberOfClaims': 4,
            'PolicyNumber': 999,
            'DayOfWeekClaimed': "Monday",
            'AgentType': "Type3",
            'RepNumber': 999,
            'Month': "December",
            'Year': 2023,
            'DriverRating': 1,
            'AccidentArea': "Urban",
            'WeekOfMonth': 1,
            'MonthClaimed': "December",
            'Make': "BMW",
            'WeekOfMonthClaimed': 1,
            'DayOfWeek': "Monday"
        }
    }
    return test_cases.get(case_name, None)

# ---------------------------
# Function to get user features
# ---------------------------
def user_input_features():
    st.sidebar.write("### Load Test Case Inputs")
    test_case = st.sidebar.selectbox("Select a Test Case", ["None", "High Fraud Risk 1"])
    case_data = load_test_case(test_case) if test_case != "None" else None

    # Helper function to set default values
    def val(key, default):
        return default if case_data is None else case_data[key]

    # Categorical
    Sex = st.selectbox("Sex", ["Male", "Female"], index=["Male","Female"].index(val('Sex', 'Male')))
    VehicleCategory = st.selectbox("Vehicle Category", ["Sedan", "SUV", "Truck"],
                                   index=["Sedan","SUV","Truck"].index(val('VehicleCategory','Sedan')))
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"],
                                 index=["Single","Married"].index(val('MaritalStatus','Single')))
    Fault = st.selectbox("Fault", ["Driver", "Company"], index=["Driver","Company"].index(val('Fault','Driver')))
    PolicyType = st.selectbox("Policy Type", ["Basic", "Premium"],
                              index=["Basic","Premium"].index(val('PolicyType','Basic')))
    PoliceReportFiled = st.selectbox("Police Report Filed", ["Yes", "No"],
                                     index=["Yes","No"].index(val('PoliceReportFiled','Yes')))
    WitnessPresent = st.selectbox("Witness Present", ["Yes", "No"],
                                  index=["Yes","No"].index(val('WitnessPresent','Yes')))
    AddressChange_Claim = st.selectbox("Address Changed After Claim", ["Yes", "No"],
                                       index=["Yes","No"].index(val('AddressChange_Claim','No')))
    AgentType = st.selectbox("Agent Type", ["Type1", "Type2", "Type3"],
                             index=["Type1","Type2","Type3"].index(val('AgentType','Type1')))
    AccidentArea = st.selectbox("Accident Area", ["Urban", "Rural"],
                                index=["Urban","Rural"].index(val('AccidentArea','Urban')))
    Make = st.selectbox("Make of Vehicle", ["Toyota", "Honda", "Ford", "BMW"],
                        index=["Toyota","Honda","Ford","BMW"].index(val('Make','Toyota')))
    DayOfWeekClaimed = st.selectbox("Day of Week Claim",
                                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                    index=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(
                                        val('DayOfWeekClaimed','Monday')))
    Month = st.selectbox("Month",
                         ["January", "February", "March", "April", "May", "June", "July", "August",
                          "September", "October", "November", "December"],
                         index=["January", "February", "March", "April", "May", "June", "July", "August",
                                "September", "October", "November", "December"].index(val('Month','January')))
    MonthClaimed = st.selectbox("Month Claimed",
                                ["January", "February", "March", "April", "May", "June", "July", "August",
                                 "September", "October", "November", "December"],
                                index=["January", "February", "March", "April", "May", "June", "July", "August",
                                       "September", "October", "November", "December"].index(val('MonthClaimed','January')))
    DayOfWeek = st.selectbox("Day of Week",
                             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                             index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(val('DayOfWeek','Monday')))
    BasePolicy = st.selectbox("Base Policy", ["Liability", "Collision", "All Perils"],
                              index=["Liability","Collision","All Perils"].index(val('BasePolicy','Liability')))

    # Numeric
    Age = st.number_input("Age", min_value=18, max_value=100, value=val('Age',30))
    VehiclePrice = st.number_input("Vehicle Price", min_value=1000, max_value=100000, value=val('VehiclePrice',20000))
    AgeOfVehicle = st.number_input("Age of Vehicle", min_value=0, max_value=50, value=val('AgeOfVehicle',5))
    AgeOfPolicyHolder = st.number_input("Age of Policy Holder", min_value=18, max_value=100, value=val('AgeOfPolicyHolder',30))
    Days_Policy_Accident = st.number_input("Days Since Accident", min_value=0, value=val('Days_Policy_Accident',30))
    Days_Policy_Claim = st.number_input("Days Since Claim", min_value=0, value=val('Days_Policy_Claim',20))
    NumberOfSuppliments = st.number_input("Number of Supplements", min_value=0, value=val('NumberOfSuppliments',0))
    NumberOfCars = st.number_input("Number of Cars", min_value=1, value=val('NumberOfCars',1))
    Deductible = st.number_input("Deductible", min_value=100, max_value=5000, value=val('Deductible',500))
    PastNumberOfClaims = st.number_input("Past Number of Claims", min_value=0, value=val('PastNumberOfClaims',0))
    DriverRating = st.number_input("Driver Rating", min_value=1, max_value=5, value=val('DriverRating',3))
    WeekOfMonth = st.number_input("Week of Month", min_value=1, max_value=4, value=val('WeekOfMonth',2))
    WeekOfMonthClaimed = st.number_input("Week of Month Claimed", min_value=1, max_value=4, value=val('WeekOfMonthClaimed',1))
    Year = st.number_input("Year", min_value=2000, max_value=2100, value=val('Year',2022))
    PolicyNumber = st.number_input("Policy Number", min_value=1, max_value=999999, value=val('PolicyNumber',101))
    RepNumber = st.number_input("Rep Number", min_value=1, max_value=9999, value=val('RepNumber',10))

    # DataFrame for model
    features = pd.DataFrame({
        'Sex': [Sex],
        'Age': [Age],
        'VehiclePrice': [VehiclePrice],
        'VehicleCategory': [VehicleCategory],
        'MaritalStatus': [MaritalStatus],
        'Fault': [Fault],
        'PolicyType': [PolicyType],
        'AgeOfVehicle': [AgeOfVehicle],
        'AgeOfPolicyHolder': [AgeOfPolicyHolder],
        'Days_Policy_Accident': [Days_Policy_Accident],
        'Days_Policy_Claim': [Days_Policy_Claim],
        'PoliceReportFiled': [PoliceReportFiled],
        'WitnessPresent': [WitnessPresent],
        'NumberOfSuppliments': [NumberOfSuppliments],
        'AddressChange_Claim': [AddressChange_Claim],
        'NumberOfCars': [NumberOfCars],
        'Deductible': [Deductible],
        'BasePolicy': [BasePolicy],
        'PastNumberOfClaims': [PastNumberOfClaims],
        'PolicyNumber': [PolicyNumber],
        'DayOfWeekClaimed': [DayOfWeekClaimed],
        'AgentType': [AgentType],
        'RepNumber': [RepNumber],
        'Month': [Month],
        'Year': [Year],
        'DriverRating': [DriverRating],
        'AccidentArea': [AccidentArea],
        'WeekOfMonth': [WeekOfMonth],
        'MonthClaimed': [MonthClaimed],
        'Make': [Make],
        'WeekOfMonthClaimed': [WeekOfMonthClaimed],
        'DayOfWeek': [DayOfWeek],
    })
    return features

# ---------------------------
# App Title
# ---------------------------
st.title("Fraudulent Claim Prediction")

# Get input data
user_data = user_input_features()

# Fraud sensitivity threshold
threshold = st.slider("Fraud Alert Threshold", 0.1, 0.9, 0.5, 0.05)

# Prediction
if st.button("Predict"):
    try:
        proba = model.predict_proba(user_data)[:, 1]
        prediction = (proba >= threshold).astype(int)
        st.subheader("Prediction Result")
        st.write(f"Predicted Fraudulent Claim: {'YES' if prediction[0] == 1 else 'NO'}")
        st.write(f"Fraud Probability: {proba[0]:.2f}")
        st.markdown("### Input Data Used for Prediction")
        st.write(user_data)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
