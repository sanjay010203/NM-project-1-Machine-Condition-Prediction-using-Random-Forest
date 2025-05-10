

# Machine Condition Prediction using Random Forest

**Sanjay U**
2nd Year, Mechanical Engineering
ARM College of Engineering & Technology
Course: Data Analysis in Mechanical Engineering

---

## Project Overview

This project focuses on predicting the working condition of industrial machines using a **Random Forest Classifier**. The goal is to analyze input data such as temperature, vibration, RPM, oil quality, and other machine-related parameters to determine whether the machine is functioning normally or showing signs of failure.

This type of prediction can be useful in maintenance planning and early fault detection, especially in mechanical and manufacturing industries.

---

## Getting Started

Before running the prediction script, make sure to install all required Python libraries. You can do this by running the command:

```bash
pip install -r requirements.txt
```

---

## Important Files

You will need the following files in your project directory:

* **`random_forest_model.pkl`**: This is the trained Random Forest model.
* **`scaler.pkl`**: This file contains the scaler used to normalize feature values during training.
* **`selected_features.pkl`**: A list of features used for training, to ensure input features are in the correct order.

Make sure these files are available before attempting any predictions.

---

## How the Prediction Works

Here is a simple breakdown of how the system makes predictions:

1. **Loading the Model and Tools**

   * Load the trained model using `joblib.load('random_forest_model.pkl')`.
   * Load the feature scaler with `joblib.load('scaler.pkl')`.
   * Load the list of selected features with `joblib.load('selected_features.pkl')`.

2. **Preparing Input Data**

   * Create a single-row DataFrame with the same features that were used during training.
   * Double-check that all feature names match exactly.

3. **Scaling the Input**

   * Normalize the data using the loaded scaler so it fits the same scale as the training data.

4. **Making Predictions**

   * Use `.predict()` to get the final class (e.g., normal or faulty).
   * Use `.predict_proba()` to understand the confidence levels of the prediction.

---

## Sample Prediction Script

Here is a basic example you can use to test the prediction process:

```python
import joblib
import pandas as pd

# Load saved model, scaler, and features
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input data (replace with real values)
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Ensure feature order matches training
new_data = new_data[selected_features]

# Normalize data
scaled_data = scaler.transform(new_data)

# Predict condition
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Condition:", prediction[0])
print("Confidence Scores:", prediction_proba[0])
```

---

## Important Notes

* Make sure to include **all features used during training** in the exact same format.
* Inputs should be within a realistic range (close to the training data values).
* **Do not change the order** of the features in the DataFrame.

---

## Updating the Model

If you want to retrain the model with new data:

* Apply the same preprocessing steps used in this project.
* Use the same feature set and scaling methods.
* Save the updated model and tools again using `joblib`.

---

## Real-World Applications

* Detecting early faults in machinery on the production floor.
* Improving maintenance scheduling in industries.
* Integrating with IoT sensors for automated condition monitoring.

