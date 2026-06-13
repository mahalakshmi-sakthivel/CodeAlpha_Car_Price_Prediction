import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress all background system and version warnings
import warnings
warnings.filterwarnings('ignore')

# Page settings
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

st.title("🚗 Car Price Prediction Dashboard")
st.markdown("Developed for **CodeAlpha Data Science Internship**. Built using **Pandas, Scikit-learn, and Matplotlib**.")

try:
    # Modified to look for the exact file name format from your data information
    df = pd.read_csv('car data.csv')
    df.columns = df.columns.str.strip() # Clean hidden trailing spaces in columns
    
    st.subheader("📋 1. Dataset Preview (Pandas DataFrame)")
    st.dataframe(df.head(), use_container_width=True)
    
    # Feature Engineering & Preprocessing
    # Auto-detect target column naming schemes (Selling_Price)
    target_col = 'Selling_Price' if 'Selling_Price' in df.columns else 'Price'
    
    # Drop unique string identifiers like Car_Name to prevent model overfitting
    if 'Car_Name' in df.columns:
        df_clean = df.drop(columns=['Car_Name'])
    else:
        df_clean = df.copy()
        
    # Convert categorical string variables into numeric dummy columns (One-Hot Encoding)
    df_encoded = pd.get_dummies(df_clean, drop_first=True)
    
    # Separate Features (X) and Target (y)
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    # Split into 80% Train and 20% Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling for fair Linear Regression weight assignment
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the Regression Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Model Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Model Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # UI Layout Split
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📊 2. Regression Model Evaluation")
        st.metric(label="R-squared (R² Score / Prediction Accuracy)", value=f"{r2 * 100:.2f}%")
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f} Lakhs")
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
        
        st.info("""
        💡 **Real-World Application:** Automotive platforms use regression algorithms to instantly evaluate a car's fair valuation. By analyzing numerical trends like depreciation (Year), mileage (Kms_Driven), and luxury features (Present_Price), businesses automate buying price estimates.
        """)

    with col_right:
        st.subheader("🔮 3. Interactive Pricing Sandbox")
        st.write("Adjust the features to predict a used car price live:")
        
        # Build dynamic sliders based on dataset thresholds
        yr_min, yr_max = int(df_clean['Year'].min()), int(df_clean['Year'].max())
        year_in = st.slider("Vehicle Model Year", yr_min, yr_max, yr_max)
        
        pr_max = float(df_clean['Present_Price'].max())
        present_in = st.slider("Current Showroom Price (in Lakhs)", 0.0, pr_max, float(df_clean['Present_Price'].mean()))
        
        kms_max = int(df_clean['Kms_Driven'].max())
        kms_in = st.slider("Total Kilometers Driven", 0, kms_max, int(df_clean['Kms_Driven'].mean()))
        
        # Construct prediction input matrix mapping structure perfectly
        user_row = pd.DataFrame(0, index=[0], columns=X.columns)
        if 'Year' in user_row.columns: user_row['Year'] = year_in
        if 'Present_Price' in user_row.columns: user_row['Present_Price'] = present_in
        if 'Kms_Driven' in user_row.columns: user_row['Kms_Driven'] = kms_in
        
        # Predict price
        user_scaled = scaler.transform(user_row)
        pred_val = max(0.0, model.predict(user_scaled)[0])
        st.success(f"💰 **Estimated Selling Price Valuation:** {pred_val:.2f} Lakhs")

    st.markdown("---")
    
    # Visualizing Results using Matplotlib
    st.subheader("📈 4. Visualizing Results (Matplotlib)")
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.scatter(y_test, y_pred, alpha=0.7, color='dodgerblue', label='Predicted Points')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit Line')
    plt.title('Linear Regression: Actual Market Price vs Model Predictions')
    plt.xlabel('Actual Price (Lakhs)')
    plt.ylabel('Predicted Price (Lakhs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

except FileNotFoundError:
    st.error("⚠️ Data File Error: Please verify that your file is named exactly 'car data.csv' and uploaded directly inside your active repository.")
