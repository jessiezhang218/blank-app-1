import streamlit as st
import pandas as pd

st.title("Wine Quality Prediction App")

df = pd.read_csv("winequality-red.csv")
st.write("Dataset preview:")
st.dataframe(df.head())
