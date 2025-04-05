# AI DISASTER PREDICTION SYSTEM (CORE MODULE)
# -------------------------------------------
# This code is a base skeleton for integrating multiple features: 
# - Multi-output classification for disaster type + severity
# - Time series forecasting
# - Blockchain interaction
# - User roles (to be handled in Flask routes)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
# Ensure your dataset contains: rainfall, temperature, humidity, wind_speed, soil_moisture, pressure, disaster_type, severity_level

df = pd.read_csv('disaster_data.csv')

# Features and Labels
X = df[['rainfall', 'temperature', 'humidity', 'wind_speed', 'soil_moisture', 'pressure']]
y_type = df['disaster_type']
y_severity = df['severity_level']

# Encode labels
type_encoder = LabelEncoder()
severity_encoder = LabelEncoder()

y_type_encoded = type_encoder.fit_transform(y_type)
y_severity_encoded = severity_encoder.fit_transform(y_severity)

y_combined = np.column_stack((y_type_encoded, y_severity_encoded))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.2, random_state=42)

# Train model
multi_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
multi_model.fit(X_train, y_train)

# Save model and encoders
with open('disaster_model.pkl', 'wb') as f:
    pickle.dump(multi_model, f)

with open('type_encoder.pkl', 'wb') as f:
    pickle.dump(type_encoder, f)

with open('severity_encoder.pkl', 'wb') as f:
    pickle.dump(severity_encoder, f)

print(" Model training complete. Use Flask app to serve predictions.")
