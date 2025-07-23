import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="Salary Class Prediction", layout="centered")

# Page title
st.title("💼 Employee Income Classification App")
st.markdown("Predict whether an employee earns **>50K or <=50K** using multiple machine learning models.")

# Load scaler
try:
    scaler = joblib.load("models/scaler.pkl")
except:
    st.error("❌ Could not load scaler. Make sure 'models/scaler.pkl' exists.")

# Load models
model_names = ["Random_Forest", "Decision_Tree", "Gradient_Boosting", "KNN", "Neural_Network"]
models = {}
for name in model_names:
    try:
        models[name] = joblib.load(f"models/{name}.pkl")
    except:
        st.warning(f"⚠️ Could not load {name}.pkl")

# Sidebar input
st.sidebar.header("🧾 Employee Features (Encoded Inputs)")
feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                 'marital-status', 'occupation', 'relationship', 'race', 'gender',
                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

user_input = []
user_input.append(st.sidebar.slider("Age", 18, 70, 30))
user_input.append(st.sidebar.selectbox("Workclass (encoded)", list(range(9))))
user_input.append(st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 500000, 250000))
user_input.append(st.sidebar.selectbox("Education (encoded)", list(range(16))))
user_input.append(st.sidebar.slider("Educational Num", 1, 16, 10))
user_input.append(st.sidebar.selectbox("Marital Status (encoded)", list(range(7))))
user_input.append(st.sidebar.selectbox("Occupation (encoded)", list(range(15))))
user_input.append(st.sidebar.selectbox("Relationship (encoded)", list(range(6))))
user_input.append(st.sidebar.selectbox("Race (encoded)", list(range(5))))
user_input.append(st.sidebar.selectbox("Gender (encoded)", [0, 1]))
user_input.append(st.sidebar.number_input("Capital Gain", 0, 99999, 0))
user_input.append(st.sidebar.number_input("Capital Loss", 0, 99999, 0))
user_input.append(st.sidebar.slider("Hours per Week", 1, 99, 40))
user_input.append(st.sidebar.selectbox("Native Country (encoded)", list(range(42))))

user_input_df = pd.DataFrame([user_input], columns=feature_names)

# Only predict when user clicks button
if st.button("🔍 Predict"):
    try:
        scaled_input = scaler.transform(user_input_df)
        st.subheader("📊 Model Predictions")
        for name, model in models.items():
            label = model.predict(scaled_input)[0]
            prediction = ">50K" if label == 1 else "<=50K"
            st.success(f"**{name.replace('_', ' ')}**: {prediction}")
    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")

# Optional: Show model comparison chart
if os.path.exists("images/model_comparison.png"):
    st.subheader("📈 Model Performance")
    st.image("images/model_comparison.png", caption="Comparison of Models", use_column_width=True)
