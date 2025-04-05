import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import pickle

# Load dataset
df = pd.read_csv("disaster_data.csv")

# Features and labels
X = df[['rainfall', 'temperature', 'humidity', 'wind_speed']]
y_type = df['disaster_type']
y_severity = df['severity_level']

# Encode labels
type_encoder = LabelEncoder()
severity_encoder = LabelEncoder()

y_type_encoded = type_encoder.fit_transform(y_type)
y_severity_encoded = severity_encoder.fit_transform(y_severity)

y_combined = np.column_stack((y_type_encoded, y_severity_encoded))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.2, random_state=42)

# Train multi-output classifier
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Save model and encoders
with open('disaster_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('type_encoder.pkl', 'wb') as f:
    pickle.dump(type_encoder, f)

with open('severity_encoder.pkl', 'wb') as f:
    pickle.dump(severity_encoder, f)

print("✅ Model and encoders saved successfully.")
