import numpy as np
#OCSVM.py

from sklearn.svm import OneClassSVM
from pymongo import MongoClient
from bson.objectid import ObjectId

# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")

# Select the database and collection
db = client['HRMonitoring']
users_collection = db['Users']

def detect_anomalies_ocsvm(user_id):
    # Fetch heart rate data
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        print("User not found.")
        return

    hr_times = []
    hr_values = []
    for key in user:
        if key.startswith("HR at"):
            hr_times.append(key)
            hr_values.append(user[key])

    # Prepare data for OC-SVM (using only heart rate values)
    X = np.array(hr_values).reshape(-1, 1)

    # Train OC-SVM model
    ocsvm = OneClassSVM(nu=0.05)  # Adjust nu parameter as needed
    ocsvm.fit(X)

    # Predict outliers (anomalies)
    outliers = ocsvm.predict(X)
    anomaly_indices = np.where(outliers == -1)[0]

    # Print anomalies
    print("OCSVM:\nAnomalies detected:")
    for idx in anomaly_indices:
        print(f"At time {hr_times[idx]}: Heart rate {hr_values[idx]}")
