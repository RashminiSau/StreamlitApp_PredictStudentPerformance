import streamlit as st
st.set_page_config(page_title="Student Performance Prediction App", page_icon="📊")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load external stylesheet
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load data
data = pd.read_csv('StudentPerformanceFactors.csv')

# Add binary target column for pass/fail
pass_threshold = 70  # Define a passing threshold for exam scores
data['Pass_Fail'] = data['Exam_Score'].apply(lambda x: 1 if x >= pass_threshold else 0)

# Sidebar for page selection
page = st.sidebar.selectbox("Select Page", ["Home", "Data Visualization", "Prediction"])

# Home Page
if page == "Home":
    st.title("Welcome to the Student Performance Analysis App")
    st.image('im8.gif', use_column_width=True)
    st.write(
        """
        This app predicts student performance based on various factors. 
        Use the Data Visualization page to explore insights in the dataset,
        or head to the Prediction page to make predictions based on your inputs.
        """
    )

# Data Visualization Page
elif page == "Data Visualization":
    # Code remains the same for visualization

# Prediction Page
elif page == "Prediction":
    st.title("Student Exam Performance Prediction")

    # Define features and target variable
    target_column = 'Pass_Fail'  # Binary target: 1 for pass, 0 for fail
    feature_columns = ['Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 'Tutoring_Sessions']
    X = data[feature_columns]
    y = data[target_column]

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classification model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Calculate model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Input fields for the specific features
    st.subheader("Enter student data:")

    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
    hours_studied = st.number_input("Hours Studied per Week", min_value=0.0, max_value=50.0, step=0.5)
    previous_scores = st.number_input("Previous Scores (out of 100)", min_value=0.0, max_value=100.0, step=1.0)
    access_to_resources = st.selectbox("Access to Resources", ['Low', 'Medium', 'High'])
    tutoring_sessions = st.number_input("Number of Tutoring Sessions per Week", min_value=0, max_value=7, step=1)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Attendance': [attendance],
        'Hours_Studied': [hours_studied],
        'Previous_Scores': [previous_scores],
        'Access_to_Resources': [access_to_resources],
        'Tutoring_Sessions': [tutoring_sessions]
    })

    # One-hot encode input data and align with training data columns
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    # Make prediction
    if st.button("Predict"):
        # Predict the pass/fail outcome
        predicted_class = model.predict(input_data)[0]
        
        # Determine pass or fail based on prediction
        grade = "🎉 Pass 🎉" if predicted_class == 1 else "❌ Fail ❌"
        
        st.write(f"Predicted Outcome: {grade}")
        
        # Display GIF based on grade
        if predicted_class == 1:
            st.image('im9.gif', width=300)  # Replace with your pass GIF file path
        else:
            st.image('im2.gif', width=300)  # Replace with your fail GIF file path
