import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv(r"D:\traffic\Traffic-intelligence\TrafficTelligence-Advanced-Traffic-Volume-Estimation-with-Machine-Learning-1\traffic_volume.csv")

# Quick check of dataset structure
print("Dataset Preview:\n", data.head())


# Initialize LabelEncoders
weather_encoder = LabelEncoder()
holiday_encoder = LabelEncoder()

# Encode categorical columns
data['weather_encoded'] = weather_encoder.fit_transform(data['weather'])
data['holiday_encoded'] = holiday_encoder.fit_transform(data['holiday'])

# Define features and target
X = data[['weather_encoded', 'holiday_encoded']].values
y = data['traffic_volume'].values

# Train model (Linear Regression)
model = LinearRegression()
model.fit(X, y)

# Save trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save encoders as dictionary
encoders = {
    'weather': weather_encoder,
    'holiday': holiday_encoder
}

with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoders, file)

print("model.pkl and encoder.pkl saved successfully.")
