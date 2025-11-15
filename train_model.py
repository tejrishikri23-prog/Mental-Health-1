import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Generate synthetic data
np.random.seed(42)
size = 600

data = pd.DataFrame({
    'age': np.random.randint(15, 60, size),
    'sleep_hours': np.random.uniform(4, 10, size),
    'social_interaction': np.random.randint(0, 7, size),
    'work_stress': np.random.randint(1, 10, size),
    'physical_activity': np.random.randint(0, 6, size),
    'mood_score': np.random.randint(1, 10, size)
})

# Create stress level label
score = (data['work_stress'] * 0.5) + (10 - data['mood_score']) + (6 - data['physical_activity'])
conditions = [score < 8, score < 14, score >= 14]
choices = ['low', 'medium', 'high']
data['stress_level'] = np.select(conditions, choices)

# Save synthetic dataset
data.to_csv("data/synthetic_data.csv", index=False)

# Train-test split
X = data.drop('stress_level', axis=1)
y = data['stress_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save trained model
with open("mental_health_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete. Model saved as mental_health_model.pkl")
