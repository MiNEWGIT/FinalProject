from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")
db = client['HRMonitoring']
users_collection = db['Users']

def detect_anomalies(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        print("User not found")
        return

    # Extract HR data and labels
    hr_data = []
    hr_times = []
    labels = []  # Ground truth labels

    for key in user:
        if key.startswith("HR at"):
            try:
                hr_data.append(int(user[key]))
                hr_times.append(key)
                # Assuming the label is stored in a field like 'label' for each HR entry
                labels.append(user.get(f"{key}_label", 1))  # Default to 1 if no label present
            except ValueError:
                print(f"Non-numeric value found for {key}: {user[key]}. Skipping this entry.")
                continue

    if len(hr_data) == 0:
        print("No HR data available for this user.")
        return

    # Convert HR data to numpy array
    hr_data = np.array(hr_data).reshape(-1, 1)
    labels = np.array(labels)

    # Initialize and fit Isolation Forest
    contamination_rate = 0.1
    iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
    iso_forest.fit(hr_data)

    # Predict anomalies
    predictions = iso_forest.predict(hr_data)
    # Convert predictions to binary labels: 1 (normal), 0 (anomaly)
    predictions_binary = (predictions == -1).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions_binary)
    print(f"Model accuracy: {accuracy:.2f}")

    # Display anomalies
    anomaly_indices = np.where(predictions == -1)[0]
    if len(anomaly_indices) > 0:
        print("Isolation Forsests:\nAnomalies detected at the following times:")
        for idx in anomaly_indices:
            anomaly_time = hr_times[int(idx)]
            print(f"{anomaly_time}: {user[anomaly_time]}")
    else:
        print("No anomalies detected.")
