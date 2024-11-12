import streamlit as st
st.set_page_config(page_title="Student Performance Prediction App", page_icon="📊")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load external stylesheet
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load data
data = pd.read_csv('StudentPerformanceFactors.csv')

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
    st.title("Data Visualization")
    st.write("Explore various visualizations of the student performance dataset.")

    # Sidebar for selecting plot type
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Correlation Heatmap", "Bar Chart", "Pie Chart", "Line Chart", "Boxplot", "Scatter Plot", "Histogram"])

    # Dropdown for feature selection based on the plot type
    if plot_type in ["Bar Chart", "Pie Chart", "Line Chart", "Boxplot", "Histogram"]:
        feature = st.sidebar.selectbox("Select Feature", data.columns)

    elif plot_type == "Scatter Plot":
        x_feature = st.sidebar.selectbox("Select X-axis Feature", data.columns)
        y_feature = st.sidebar.selectbox("Select Y-axis Feature", data.columns)

    # Plotting based on selected plot type
    st.subheader(f"{plot_type} Visualization")

    # Correlation Heatmap
    if plot_type == "Correlation Heatmap":
        st.write("Correlation Heatmap of the Dataset")
        
        # Select only numerical columns for correlation calculation
        numerical_data = data.select_dtypes(include=[np.number])
        corr = numerical_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Bar Chart
    elif plot_type == "Bar Chart":
        st.write(f"Bar Chart of {feature}")
        fig, ax = plt.subplots()
        data[feature].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Pie Chart
    elif plot_type == "Pie Chart":
        st.write(f"Pie Chart of {feature}")
        fig, ax = plt.subplots()
        data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)

    # Line Chart
    elif plot_type == "Line Chart":
        st.write(f"Line Chart of {feature}")
        fig, ax = plt.subplots()
        data[feature].plot(kind='line', ax=ax)
        ax.set_xlabel("Index")
        ax.set_ylabel(feature)
        st.pyplot(fig)

    # Boxplot
    elif plot_type == "Boxplot":
        st.write(f"Boxplot of {feature}")
        fig, ax = plt.subplots()
        sns.boxplot(data=data, y=feature, ax=ax)
        st.pyplot(fig)

    # Scatter Plot
    elif plot_type == "Scatter Plot":
        st.write(f"Scatter Plot between {x_feature} and {y_feature}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_feature, y=y_feature, ax=ax)
        st.pyplot(fig)

    # Histogram
    elif plot_type == "Histogram":
        st.write(f"Histogram of {feature}")
        fig, ax = plt.subplots()
        data[feature].plot(kind='hist', bins=20, ax=ax)
        ax.set_xlabel(feature)
        st.pyplot(fig)

# Prediction Page
elif page == "Prediction":
    st.title("Student Exam Score Prediction")

    # Define features and target variable
    target_column = 'Exam_Score'  # Replace with the actual name of your exam score column
    feature_columns = ['Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 'Tutoring_Sessions']
    X = data[feature_columns]
    y = data[target_column]

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train regression model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Calculate model accuracy
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"Model RMSE: {rmse:.2f}")

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
        # Predict the exam score
        predicted_score = model.predict(input_data)[0]
        
        # Determine pass or fail (assuming a passing score is 70)
        grade = "🎉 Pass 🎉" if predicted_score >= 75 else "❌ Fail ❌"
        
        st.write(f"Predicted Exam Score: {predicted_score:.2f}")
        st.write(f"Grade: {grade}")
        
       # Display GIF based on grade
        if predicted_score >= 75:
            st.image('im9.gif', width=300)  # Replace with your pass GIF file path
        else:
            st.image('im2.gif', width=300)  # Replace with your fail GIF file path
            
