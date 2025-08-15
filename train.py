import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load data
df = pd.read_excel('fraud_oracle.xlsx')

# Features and target
target_col = 'FraudFound_P'
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

# Handle mixed-type columns
# For categorical columns, convert all object columns to strings
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Ensure all categorical columns are strings
for col in categorical_cols:
    X[col] = X[col].astype(str)

# Ensure all numeric columns are in a proper numeric type
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Handle missing values by filling NaN entries
X.fillna(method='ffill', inplace=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define transformers for numeric and categorical columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, class_weight='balanced', n_jobs=-1)

# Combine preprocessing and model into a pipeline
pipe = Pipeline(steps=[('prep', preprocessor), ('rf', clf)])

# Train the model
pipe.fit(X_train, y_train)

# Evaluate the model
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_prob),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
}


# Save the model and metrics
import os
os.makedirs('models', exist_ok=True)
model_path = 'models/fraud_rf.pkl'
metrics_path = 'models/metrics.json'
joblib.dump(pipe, model_path)

# Save the metrics
import json
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Model saved to {model_path}")
print(f"Metrics saved to {metrics_path}")
