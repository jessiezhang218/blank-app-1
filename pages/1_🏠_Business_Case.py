import streamlit as st
import pandas as pd

# Page title
st.title("🍷 Wine Quality Prediction – Business Case & Dataset")

# Description
st.header("Business Problem")
st.write("""
Wineries need a reliable way to **predict wine quality** before bottling or selling.
By analyzing the wine’s chemical properties, we can estimate its quality score.
This helps producers improve quality control, adjust production, and set pricing strategies.
""")

# Load dataset
df = pd.read_csv("winequality-red.csv")

st.header("Dataset Overview")
st.write("This dataset contains physicochemical test results for red 'Vinho Verde' wine from Portugal.")

# Show preview
st.dataframe(df.head())

# Describe columns
st.markdown("""
**Features (inputs):**
- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  

**Target (output):**
- Quality (score between 0 and 10)
""")

st.caption("Source: Cortez et al., 2009 – UCI Machine Learning Repository")
