import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


st.set_page_config(page_title="Wine Quality Prediction App", page_icon="🍷")

st.title("🍷 Wine Quality Prediction App")
st.markdown("""
Welcome to our app!  
Navigate through the sidebar to explore our project:
- **Landing Page** – Learn about the problem and dataset  
- **Visualization Page** – Explore key insights  
- **Prediction Page** – Predict wine quality using Linear Regression  
""")
df = pd.read_csv("winequality-red.csv")
st.sidebar.title("Wine")
page = st.sidebar.selectbox("Select Page",["Introduction","Visualization","Prediction"])

st.write("   ")


if page == "Introduction":
    st.subheader("01 Introduction")
    st.markdown("Include preview on your topic here.")


    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing Values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values found!")
    else:
        st.warning("You have missing values!")

    st.markdown("##### Summary Statistics")
    if st.toggle("Show Describe Table"):
        st.dataframe(df.describe())


elif page == "Visualization":
    st.subheader("02 Visualization")

    st.markdown("###### Quality distribution")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.countplot(x="quality", data=df, ax=ax1)
    ax1.set_xlabel("Quality (score)")
    ax1.set_ylabel("Count")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)
    st.pyplot(fig1)

    st.markdown("###### Correlation heatmap")
    corr = df.corr(numeric_only=True)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3, ax=ax2)
    st.pyplot(fig2)

    st.markdown("###### Alcohol vs. Quality (with jitter)")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    sns.stripplot(x="quality", y="alcohol", data=df, alpha=0.6, ax=ax3)
    ax3.set_ylabel("Alcohol")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.3)
    st.pyplot(fig3)

elif page == "Prediction":
    st.subheader("03 Prediction Modeling")

    # features/target
    target_col = "quality"
    feature_cols = [c for c in df.columns if c != target_col]

    st.markdown("**Select features for the model**")
    chosen_features = st.multiselect(
        "Features",
        options=feature_cols,
        default=feature_cols,  # all by default
    )

    if len(chosen_features) == 0:
        st.info("Please select at least one feature.")
        st.stop()

    X = df[chosen_features]
    y = df[target_col]

    test_size = st.slider("Test size (%)", 10, 40, 20, step=5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100.0, random_state=42
    )

    # model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("#### Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
    c2.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")
    c3.metric("R² Score", f"{r2:.3f}")

    # scatter: Actual vs Predicted, hue = |error|
    plot_df = y_test.reset_index(drop=True).to_frame(name="Actual")
    plot_df["Predicted"] = y_pred
    plot_df["Error"] = (plot_df["Actual"] - plot_df["Predicted"]).abs()

    st.markdown("#### Actual vs Predicted (Hue = |Error|)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="Actual",
        y="Predicted",
        hue="Error",
        palette="plasma",
        edgecolor=None,
        ax=ax,
        legend=True,
    )
    line_min = float(plot_df[["Actual", "Predicted"]].min().min())
    line_max = float(plot_df[["Actual", "Predicted"]].max().max())
    ax.plot([line_min, line_max], [line_min, line_max], "--", color="gray")
    ax.set_xlabel("Actual Quality")
    ax.set_ylabel("Predicted Quality")
    ax.set_title("Actual vs Predicted Wine Quality")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="|Error|")
    st.pyplot(fig)

    st.markdown(
        "###### Notes\n"
        "- Linear Regression provides a simple baseline; a modest R² is common on this dataset.\n"
        "- Consider regularization (Ridge/Lasso), tree ensembles, or non-linear models for better accuracy.\n"
        "- Feature engineering (e.g., interactions, log transforms) can further improve performance."
    )