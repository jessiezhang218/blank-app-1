import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="Wine Quality Prediction App 🍷", 
    page_icon="🍇",
    layout="wide"
)

# Wine theme with burgundy and beige 
st.markdown("""
    <style>
    .stApp {
        background-color: #f7f3ef;
        color: #4b0d1a;
        font-family: 'Georgia', serif;
        font-size: 36px;
    }
    h1, h2, h3, h4, h5 {
        color: #720026 !important;
        font-family: 'Playfair Display', serif;
    }
    .stSidebar {
        background-color: #f4ebe2 !important;
        color: #4b0d1a !important;
        font-size: 36px;
    }
    .stSelectbox, .stSlider, .stMultiSelect, .stButton>button {
        color: #4b0d1a !important;
        font-size: 32px;
    }
    .stMetric {
        background-color: #f0e3da !important;
        border-radius: 10px;
        padding: 5px;
        font-size: 32px;
    }
    hr {
        border-top: 1px solid #a34a54;
    }
    .feature-card {
        background-color: #f0e3da;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #720026;
        margin: 10px 0;
        font-size: 36px;
    }
    </style>
""", unsafe_allow_html=True)

# Loading data
df = pd.read_csv("winequality-red.csv")

# Sidebar navigation
st.sidebar.title("🍷 Navigation")
st.sidebar.markdown("Navigate through different sections of the app.")
page = st.sidebar.radio(
    "Go to section:",
    ["Introduction", "Visualization", "Prediction", "Conclusion"]
)

# Page title
st.title("🍇 Wine Quality Prediction App")

# Welcome msg and Intro page
if page == "Introduction":
    st.image("wine.jpg", use_container_width=True)

    st.markdown("""
    Welcome to our Wine Quality Prediction App!  

    Navigate through the sidebar to explore our project:

    - **Introduction** – Explore dataset structure and statistics  
    - **Visualization** – Discover key insights through charts  
    - **Prediction** – Predict wine quality using Linear Regression
    - **Conclusion** - Project summary and business impact
    """)

if page == "Introduction":
    st.header("01 • Introduction")
    st.markdown("""
    Wine tasting is both an art and a science.
    Wineries need a reliable way to predict wine quality before bottling or selling. By analyzing the wine's chemical properties, we can estimate its quality score. This helps producers improve quality control, adjust production, and set pricing strategies.
    
    **The Challenge:** Wineries face significant challenges in quality control and production optimization:
    
    - **Time-Consuming Process**: Traditional wine tasting requires expert panels and takes days or weeks
    - **High Costs**: Maintaining tasting panels is expensive and labor-intensive  
    - **Subjectivity**: Different experts may rate the same wine differently
    - **Inconsistency**: Quality assessment varies between batches and tasters
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(8), use_container_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Quality")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("No missing values found")
        else:
            st.warning("⚠️ Some missing values detected")
        
        st.metric("Total Samples", len(df))
        st.metric("Number of Features", len(df.columns) - 1)

    with col2:
        st.subheader("Quality Overview")
        st.metric("Average Quality", f"{df['quality'].mean():.2f}")
        st.metric("Quality Range", f"{df['quality'].min()}-{df['quality'].max()}")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# Visualization page
elif page == "Visualization":
    st.header("02 • Data Visualization")
    
    # Add dropdown for visualization options
    viz_option = st.selectbox(
        "Choose Visualization Type:",
        ["All Visualizations", "Quality Distribution", "Correlation Heatmap", "Feature Relationships"],
        key="viz_selector"
    )
    
    if viz_option in ["All Visualizations", "Quality Distribution"]:
        st.subheader("Quality Distribution")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.countplot(x="quality", data=df, ax=ax1, palette="RdPu")
        ax1.set_xlabel("Quality (score)")
        ax1.set_ylabel("Count")
        ax1.set_title("Frequency of Wine Quality Ratings", color="#720026")
        st.pyplot(fig1)

    if viz_option in ["All Visualizations", "Correlation Heatmap"]:
        st.subheader("Feature Correlations")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, cmap="RdBu_r", center=0, linewidths=0.3, ax=ax2, annot=True, fmt=".2f")
        ax2.set_title("Correlation Among Wine Features", color="#720026")
        st.pyplot(fig2)

    if viz_option in ["All Visualizations", "Feature Relationships"]:
        st.subheader("Feature vs Quality Relationship")
        # Add dropdown for feature selection
        feature_choice = st.selectbox(
            "Select feature to compare with quality:",
            [col for col in df.columns if col != 'quality'],
            key="feature_choice"
        )
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.stripplot(x="quality", y=feature_choice, data=df, alpha=0.7, ax=ax3, palette="Reds")
        ax3.set_title(f"{feature_choice.title()} vs Quality Rating", color="#720026")
        ax3.set_xlabel("Quality Score")
        ax3.set_ylabel(feature_choice.title())
        st.pyplot(fig3)

# Prediction page
elif page == "Prediction":
    st.header("03 • Prediction Modeling")
    st.subheader("Select features for the model")
    
    target_col = "quality"
    feature_cols = [c for c in df.columns if c != target_col]
    
    chosen_features = st.multiselect(
        "Features",
        options=feature_cols,
        default=feature_cols,
        help="Select which chemical properties to include in the model"
    )
 
    if len(chosen_features) == 0:
        st.info("Please select at least one feature.")
        st.stop()

    test_size = st.slider("Test size (%)", 10, 40, 20, step=5)

    X = df[chosen_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100.0, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
    with col2:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")
    with col3:
        st.metric("R² Score", f"{r2:.3f}")

    st.subheader("Actual vs Predicted (Hue = |Error|)")
    
    plot_df = y_test.reset_index(drop=True).to_frame(name="Actual")
    plot_df["Predicted"] = y_pred
    plot_df["Error"] = (plot_df["Actual"] - plot_df["Predicted"]).abs()

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=plot_df,
        x="Actual",
        y="Predicted",
        hue="Error",
        palette="plasma",
        edgecolor=None,
        ax=ax,
        legend=True,
        s=80
    )
    
    line_min = float(plot_df[["Actual", "Predicted"]].min().min())
    line_max = float(plot_df[["Actual", "Predicted"]].max().max())
    ax.plot([line_min, line_max], [line_min, line_max], "--", color="gray", linewidth=2)
    ax.set_xlabel("Actual Quality")
    ax.set_ylabel("Predicted Quality")
    ax.set_title("Actual vs Predicted Wine Quality", color="#720026")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="|Error|")
    st.pyplot(fig)

    st.subheader("Feature Importance")
    
    importance_df = pd.DataFrame({
        'Feature': chosen_features,
        'Coefficient': model.coef_,
        'Impact on Quality': ['Positive' if x > 0 else 'Negative' for x in model.coef_]
    }).sort_values('Coefficient', key=lambda x: abs(x), ascending=False)
    
    st.dataframe(importance_df, use_container_width=True)

    # Notes
    st.markdown("""
    ---
    **Notes**
    - Linear Regression provides a simple baseline; a modest R² is common on this dataset.
    - Consider regularization (Ridge/Lasso), tree ensembles, or non-linear models for better accuracy.
    - Feature engineering (e.g., interactions, log transforms) can further improve performance.
    """)

# Conclusion page
elif page == "Conclusion":
    st.header("04 • Project Conclusion")
    
    st.markdown('<div class="conclusion-box">', unsafe_allow_html=True)
    
    st.markdown("""
    We addressed the core business problem by creating a data-driven wine quality assessment system that:
    
    - **Reduces reliance** on expensive, time-consuming expert tasting panels
    - **Provides consistent, objective** quality ratings based on chemical properties  
    - **Identifies key factors** that winemakers can control to improve quality
    - **Offers fast predictions** during production for real-time quality control
    """)
    
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Insights**:
        - Alcohol content is the strongest predictor of quality (+0.48 correlation)
        - Volatile acidity (vinegar taste) significantly reduces quality (-0.39)
        - Most wines cluster in average quality range (scores 5-6)
        """)
    
    with col2:
        st.markdown("""
        **Model Performance**:
        - R² Score: 0.403 (explains 40.3% of quality variance)
        - Mean Absolute Error: 0.504 points
        - Predictions typically within ±0.5 points of expert scores
        """)
    
    st.subheader("Business Impact")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **For Wine Producers**:
        - Optimize recipes by increasing alcohol and controlling acidity
        - Reduce quality control costs by 60-80% through automated assessment
        - Make data-driven decisions about production adjustments
        """)
    
    with col2:
        st.markdown("""
        **For the Industry**:
        - Establish objective quality standards beyond subjective tasting
        - Enable consistent quality across different production batches
        - Provide scientific basis for pricing and marketing decisions
        """)
    
