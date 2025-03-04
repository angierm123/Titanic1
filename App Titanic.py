import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open('titanic_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict if they would survive.")

# User input 
pclass = st.selectbox("Passenger class (Pclass)", [1, 2, 3])
sex = st.radio("Sex", ["Man", "Woman"])
age = st.slider("Age", 0, 100, 30)
fare = st.number_input("Rate paid (Fare)", min_value=0.0, value=32.0, step=1.0)
embarked = st.selectbox("Port of embarkation (Embarked)", ["C", "Q", "S"])

# Convert input to numeric format
sex = 1 if sex == "Man" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_mapping[embarked]

# Prepare data for prediction
input_data = np.array([[pclass, sex, age, fare, embarked]])
input_data = scaler.transform(input_data)

# Prediction
if st.button("Predict"): 
    prediction = model.predict(input_data)
    result = "He survived" if prediction[0] == 1 else "Did not survive"
    st.write(f"### result: {result}")
