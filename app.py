import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def pretty_model_summary(model):
    layers = []
    for layer in model.layers:
        layers.append({
            "Name": layer.name,
            "Type": layer.__class__.__name__,
            # "Output Shape": (None, layer.units),
            "Param #": layer.count_params()
        })

    df = pd.DataFrame(layers)
    st.dataframe(df, use_container_width=True)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
 
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae
}


def load_data():
    data = pd.read_csv(filepath_or_buffer='Loan_default.csv')
    data = data.drop(['LoanID', 'Education', 'HasDependents', 'LoanPurpose', 'HasCoSigner'], axis=1)
    return data


def load_advanced_mlp():
    model = load_model('bank_loan_default_adv_mlp.keras', safe_mode=False)
    model.summary()
    return model


def load_basic_ann():
    model = load_model('bank_loan_default_ann.keras', safe_mode=False)
    model.summary()
    return model


def load_residual_mlp():
    model = load_model('bank_loan_default_res_mlp.keras', safe_mode=False)
    model.summary()
    return model


st.set_page_config(
    page_title="Bank Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #0066cc; text-align: center; margin-bottom: 30px;}
    .prediction-result {font-size: 1.8rem; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .model-metrics {background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)


# Title
st.markdown("<h1 class='main-header'>Bank Loan Default Predictor 🏦</h1>", unsafe_allow_html=True)

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox(
                "Choose a Model",
                ("Build Basic ANN", "Build Advanced MLP", "Build Residual MLP")
)

# Loading the actual dataset before performing data exploration.
data = load_data()


st.write("### Enter Your Information")
    
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income (Rs.)", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount (Rs.)", min_value=0, value=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

with col2:
    months_employed = st.number_input("Months Employed", min_value=0, value=24)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=5)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
    loan_term = st.number_input("Loan Term (months)", min_value=1, value=60)

with col3:
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3)
    employment_type = st.selectbox("Employment Type", ["Part-time", "Self-employed", "Unemployed", "Full-time"])
    marital_status = st.selectbox("Marital Status", ["Married", "Divorced", "Single"])
    has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])


# --- Preprocessing inputs ---
employment_mapping = {"Part-time": 0, "Self-employed": 1, "Unemployed": 2, "Full-time": 3}
marital_mapping = {"Married": 1, "Divorced": 0, "Single": 2}
mortgage_mapping = {"Yes": 1, "No": 0}

input_features = np.array([[
    age,
    income,
    loan_amount,
    credit_score,
    months_employed,
    num_credit_lines,
    interest_rate,
    loan_term,
    dti_ratio,
    employment_mapping[employment_type],
    marital_mapping[marital_status],
    mortgage_mapping[has_mortgage]
]])

# --- Select and Build Model ---

if model_choice == "Build Basic ANN":
    model = load_basic_ann()
elif model_choice == "Build Advanced MLP":
    model = load_advanced_mlp()
else:
    model = load_residual_mlp()

st.subheader("Model Summary")
pretty_model_summary(model)

# --- Perform Prediction (Dummy Training for Example Purposes) ---

# NOTE: Normally, you should load trained weights. Here, it's random prediction.
predict_btn = st.button("Predict Default Risk")

if predict_btn:
    prediction = float(model.predict(input_features)[0][0])
    # st.success(f"Predicted Default Risk Probability : **{round(prediction, 4)}**")
    st.markdown(f"<div class='prediction-result' style='background-color: #0066cc;'>Predicted Default Risk Probability : {round(prediction, 4)}</div>", unsafe_allow_html=True)

    if prediction > 0.5:
        st.error("⚠️ High Risk of Loan Default!")
    else:
        st.info("✅ Low Risk of Loan Default!")


    # Making data ready for performing the metrics calculations using the desired model. 
    features = data.drop('Default', axis=1)
    labels = data['Default']

    # Mapping the data, in the same way in which the data is mapped with the user-input.
    features['MaritalStatus'] = [marital_mapping[i] for i in features['MaritalStatus']]
    features['EmploymentType'] = [employment_mapping[i] for i in features['EmploymentType']]
    features['HasMortgage'] = [mortgage_mapping[i] for i in features['HasMortgage']]

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=123)
    
    metrics = evaluate_model(model, x_test, y_test)
                        
    # Display metrics
    st.write("### Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE", f"{metrics['MSE']:.2f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col3.metric("R²", f"{metrics['R²']:.4f}")
    col4.metric("MAE", f"{metrics['MAE']:.2f}")
    


# Dataset exploration section.
if st.sidebar.checkbox("Show Dataset Information"):
    st.sidebar.write("### Dataset Overview")
    st.sidebar.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    
    # Show sample data
    if st.sidebar.checkbox("Show Sample Data"):
        st.sidebar.dataframe(data.head())
        
    # Show feature distributions
    if st.sidebar.checkbox("Show Feature Distributions"):
        feature = st.sidebar.selectbox("Select Feature", options=data.columns)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data[feature], kde=True, ax=ax)
        st.sidebar.pyplot(fig)

# About section
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application predicts whether an individual will Default to pay his/her Bank Loan using various Machine Learning Algorithms. "
    "You can select multiple models to compare their predictions and performance metrics."
)
