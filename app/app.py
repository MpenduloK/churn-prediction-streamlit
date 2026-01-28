import streamlit as st
import joblib
import json
import pandas as pd
from pathlib import Path

# App title and description
st.title("Churn Prediction App")
st.divider()
st.write("Please the following details and hit the predict button to get a prediction.")
st.divider()

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

# This pipeline includes one-hot encoding and the Logistic Regression model.
small_pipe = joblib.load(MODEL_DIR / "churn_pipe_small.joblib")

cate_path = MODEL_DIR / "categories.json"
if not cate_path.exists():
    st.error("categories.json not found.")
    st.stop()

with open(cate_path, "r", encoding="utf-8") as f:
    cate = json.load(f)

# These 4 fields will be used for the SMALL model.
state = st.selectbox("Enter State (e.g., OH)", cate["State"])
area_code = st.selectbox("Enter Area Code (e.g., 408)", [int(x) for x in cate["Area code"]])
int_plan = st.selectbox("Enter the International plan", cate["International plan"])
vc_plan = st.selectbox("Enter the Voicemail Plan", cate["Voice mail plan"])

st.divider()

# Predict APP
predictbutton = st.button("Predict!")

if predictbutton:

    # Built a single-row DataFrame with column names matching the training DataFrame.
    X_input = pd.DataFrame([{"State": state, "Area code": int(area_code), "International plan": int_plan, "Voice mail plan": vc_plan}])

    prediction = small_pipe.predict(X_input)[0]
    predicted = "Churn" if prediction == 1 else "Not Churn"
    st.write(f"Predicted: {predicted}")

else:
    st.write("Fill in the fields and click Predict.")