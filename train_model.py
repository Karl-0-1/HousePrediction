# train_model.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# For this quick project, we'll use a few key features to keep the UI simple
features = ['MedInc', 'HouseAge', 'AveRooms', 'Population']
X = X[features]

# 2. Split data (optional for this sprint, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully!")

# 4. Save the model
joblib.dump(model, 'model.joblib')
print("Model saved to model.joblib")
