import streamlit as st
import pandas as pd
import os
import joblib

from src.depression_predictor import StudentDepressionPredictor
from src.student_case_manager import StudentCaseManager
from src.data_validator import DataValidator

# === Step 1: Download model from Google Drive ===
MODEL_URL = "https://drive.google.com/uc?id=1JIa3s1eTRZvk05en7vL-r_4LFb_FF047"
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    import gdown
    st.info("Downloading model file...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# === Step 2: Load preprocessor ===
pipeline_path = os.path.join("preprocessor.pkl")
preprocessor = joblib.load(pipeline_path)

# === Step 3: Initialize predictor ===
dataset_path = os.path.join("data", "student_depression_dataset.csv")
predictor = StudentDepressionPredictor(dataset_path)
predictor.model = joblib.load(MODEL_PATH)
predictor.preprocessor = preprocessor

# === Step 4: Streamlit UI ===
st.set_page_config(page_title="Student Depression Risk Predictor", layout="centered")
st.title("🎓 Student Depression Risk Predictor")

case_manager = StudentCaseManager()
validator = DataValidator()

cases = case_manager.list_student_cases()
case_df = pd.DataFrame(cases)
st.subheader("📋 Available Student Cases")
st.dataframe(case_df, use_container_width=True)

case_numbers = case_df["Index"].tolist()
selected_case = st.selectbox("Select a student case index", options=case_numbers)

if st.button("🔍 Predict Depression Risk"):
    student_case = case_manager.get_student_case(selected_case)
    validated_case = validator.validate_input(student_case)
    if validated_case is None:
        st.error("❌ Invalid data format in student case.")
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
                st.success(f"✅ Risk Prediction: {percent:.2f}% ({level})")
            else:
                st.error("❌ Prediction failed.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
