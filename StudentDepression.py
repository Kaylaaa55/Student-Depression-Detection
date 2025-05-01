import streamlit as st
import pandas as pd
import numpy as np
import pickle
#from PIL import Image

# Konfigurasi halaman dan judul
st.set_page_config(page_title="Student Depression Detetction", layout="wide", initial_sidebar_state="auto")
st.write("""
# Student Depression Detection

Website ini dibuat oleh : [Amelia Putri Kayla](https://www.linkedin.com/in/budi-setiawan-1a0b4a1b6/).
""")

#Menambahkan Gambar dan Pengantar
#image = Image.open("Student_Depression.jpeg")
#st.image(image, caption="Student Depression")

st.markdown("""
Student Depression Prediction

Sleep duration is a crucial indicator of mental health among students.  
According to the *American Academy of Sleep Medicine*, young adults (ages 18–25) are recommended to sleep between **7 to 9 hours per night** for optimal physical and mental health.  
However, due to academic and work-related stress, many students suffer from **sleep deprivation**, which increases the risk of **stress, fatigue, and even depression**.

> **Source**: [Sleep Foundation - Sleep Guidelines](https://www.sleepfoundation.org/how-sleep-works/how-much-sleep-do-we-really-need)

---

### 🎯 Purpose of This Model

This prediction model is built to **identify whether a student is experiencing depression** based on several contributing factors, such as:

- Gender
- Profession
- Degree
- Suicidal thoughts
- Family history
- Financial stress
- Dietary habits
- Age
- Academic pressure
- Work Pressure
- CGPA
- Study satisfaction
- Job satisfaction
- Sleep duration
- Work hours

By detecting early signs of depression, this tool aims to support mental health awareness and provide actionable insights for students, educators, and mental health professionals.
""")

# Load model
#with open("StudentDepression4.pkl", "rb") as file:
    model = pickle.load(file)

@st.cache_resource
def load_model():
    with open("StudentDepression4.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Form input data
st.header("Input Student's Data:")

gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
profession = st.selectbox("Profession", ['Student', 'Civil Engineer', 'Architect', 'UX/UI Designer', 'Digital Marketer','Chef', 'Doctor', 'Lawyer', 'Entrepreneur', 'Pharmacist','Others'])
degree = st.selectbox("Degree", ['B.Pharm', 'BSc', 'BA', 'BCA', 'M.Tech', 'B.Tech', 'BBA', 'ME', 'MHM', 'Others'])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ['Yes', 'No'])
family_history = st.selectbox("Family History of Mental Illness", ['Yes', 'No'])
financial_stress = st.selectbox("Financial Stress (1=Rendah, 5=Tinggi)", [1, 2, 3, 4, 5])
dietary_habits = st.selectbox("Dietary Habits", ['Healthy', 'Moderate', 'Unhealthy', 'Others'])

age = st.slider("Age", 16, 40, 21)
academic_pressure = st.slider("Academic Pressure (1=Rendah, 5=Tinggi)", 1, 5, 3)
work_pressure = st.slider("Work Pressure (1=Rendah, 5=Tinggi)", 1, 5, 3)
cgpa = st.number_input("CGPA (Contoh: 3.5)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
study_satisfaction = st.slider("Study Satisfaction (1=Rendah, 5=Tinggi)", 1, 5, 3)
job_satisfaction = st.slider("Job Satisfaction (1=Rendah, 5=Tinggi)", 1, 5, 3)
sleep_duration = st.slider("Sleep Duration", 0, 24, 12)
work_hours = st.slider("Work/Study Hours", 0, 24, 12)

# Prediksi jika tombol ditekan
if st.button("Depression Prediction"):
    # Buat dataframe dari input pengguna
    input_data = pd.DataFrame({
        'Age': [age],
        'Academic Pressure': [academic_pressure],
        'Work Pressure': [work_pressure],
        'CGPA': [cgpa],
        'Study Satisfaction': [study_satisfaction],
        'Job Satisfaction': [job_satisfaction],
        'Sleep Duration': [sleep_duration],
        'Work/Study Hours': [work_hours],
        'Gender': [gender],
        'Profession': [profession],
        'Degree': [degree],
        'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
        'Family History of Mental Illness': [family_history],
        'Financial Stress': [financial_stress],
        'Dietary Habits': [dietary_habits]
    })

    # Prediksi dengan model
    prediction = model.predict(input_data)[0]
    prediction_label = "Depression" if prediction == 1 else "Not Depression"

    # Tampilkan hasil
    st.subheader("Prediction Result:")
    st.success(f"The student is predicted: **{prediction_label}**")
