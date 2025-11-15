import streamlit as st
import numpy as np
import pickle

st.title("Mental Health Stress Level Predictor (Bhutan)")

# Load model
model = pickle.load(open("mental_health_model.pkl", "rb"))

# Input fields
age = st.slider('Age', 15, 60, 25)
sleep_hours = st.slider('Sleep Hours per Night', 4.0, 10.0, 7.0)
social_interaction = st.slider('Days per Week with Social Interaction', 0, 7, 3)
work_stress = st.slider('Work/Study Stress Level (1-10)', 1, 10, 5)
physical_activity = st.slider('Physical Activity Days per Week', 0, 6, 2)
mood_score = st.slider('Mood Score (1 low, 10 high)', 1, 10, 6)

if st.button("Predict Stress Level"):
    features = np.array([[age, sleep_hours, social_interaction, work_stress, physical_activity, mood_score]])
    prediction = model.predict(features)[0]
    st.write("Predicted Stress Level:", prediction)
