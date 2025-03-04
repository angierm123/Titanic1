import streamlit as st
import pickle
import numpy as np

# Cargar el modelo y el scaler
with open('titanic_model(1).pkl', 'rb') as f:
    model, scaler = pickle.load(f)

st.title("Predicción de Supervivencia en el Titanic")
st.write("Ingrese los detalles del pasajero para predecir si sobreviviría.")

# Entrada de usuario
pclass = st.selectbox("Clase del pasajero (Pclass)", [1, 2, 3])
sex = st.radio("Sexo", ["Hombre", "Mujer"])
age = st.slider("Edad", 0, 100, 30)
fare = st.number_input("Tarifa pagada (Fare)", min_value=0.0, value=32.0, step=1.0)
embarked = st.selectbox("Puerto de embarque (Embarked)", ["C", "Q", "S"])

# Convertir entrada a formato numérico
sex = 1 if sex == "Hombre" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_mapping[embarked]

# Preparar datos para predicción
input_data = np.array([[pclass, sex, age, fare, embarked]])
input_data = scaler.transform(input_data)

# Predicción
if st.button("Predecir"): 
    prediction = model.predict(input_data)
    result = "Sobrevivió" if prediction[0] == 1 else "No sobrevivió"
    st.write(f"### Resultado: {result}")