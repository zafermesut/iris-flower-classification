import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

model = load('iris_model.joblib')

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    prediction = model.predict(features)
    prediction = prediction[0]
    return prediction

st.title('Iris Flower Prediction App')

sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, step=0.5)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, step=0.5)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, step=0.5)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, step=0.5)

if st.button('Predict'):
    prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.write(f'The predicted species is {prediction}')


