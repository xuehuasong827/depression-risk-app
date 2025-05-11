
import streamlit as st
import pandas as pd
import os
import joblib

from src.depression_predictor import StudentDepressionPredictor
from src.student_case_manager import StudentCaseManager
from src.data_validator import DataValidator

# Set project paths
project_root = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(project_root, "model.pkl")
pipeline_path = os.path.join(project_root, "preprocessor.pkl")
dataset_path = os.path.join(project_root, "data", "student_depression_dataset.csv")

# Load model and preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(pipeline_path)
predictor = StudentDepressionPredictor(dataset_path)
predictor.model = model
predictor.preprocessor = preprocessor

# Initialize case manager and validator
case_manager = StudentCaseManager()
validator = DataValidator()

# Streamlit UI
st.set_page_config(page_title="Student Depression Risk Predictor", layout="centered")
st.title("🎓 Student Depression Risk Predictor")

# Display available student cases
cases = case_manager.list_student_cases()
case_df = pd.DataFrame(cases)

st.subheader("📋 Available Student Cases")
st.dataframe(case_df, use_container_width=True)

# Select a case
case_numbers = case_df["Index"].tolist()
selected_case = st.selectbox("Select a student case index to evaluate", options=case_numbers)

# Trigger prediction
if st.button("🔍 Predict Depression Risk"):
    student_case = case_manager.get_student_case(selected_case)
    if student_case:
        validated_case = validator.validate_input(student_case)
        if validated_case is None:
            st.error("❌ Data validation failed. Please check the student case fields.")
        else:
            try:
                prediction = predictor.predict_depression(validated_case, selected_case)
                if prediction is not None:
                    percent = prediction[0] * 100
                    if percent < 30:
                        level = "LOW"
                    elif percent < 60:
                        level = "MEDIUM"
                    else:
                        level = "HIGH"
                    st.success(f"✅ Prediction complete: Risk = {percent:.2f}% ({level})")
                else:
                    st.error("❌ Prediction could not be completed.")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
