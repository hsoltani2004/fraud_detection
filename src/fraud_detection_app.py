import streamlit as st
import joblib
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")

# Custom CSS for better aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px #aaa;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    return model


model = load_model()

# Define mappings for categorical encoding
country_mapping = {"US": 0, "GB": 1, "AU": 2, "IN": 3, "DE": 4, "FR": 5}
currency_mapping = {"USD": 0, "EUR": 1, "GBP": 2, "AUD": 3, "INR": 4}
type_mapping = {"MT103": 0, "MT202": 1, "MT202C": 2}

# App title and description
st.title("🛡️ Fraud Detection System")
st.markdown("#### Enter transaction details below to predict the likelihood of fraud.")
# st.info('ℹ️ Please enter original (raw) transaction values. Log transformation and encoding will be applied automatically.')

# Sidebar for model information
st.sidebar.title("About Model")
st.sidebar.markdown("""
- **Model**: XGBoost Classifier
- **Purpose**: Predict if a transaction is Fraudulent or Normal
""")

# Input fields for features
st.markdown("### Transaction Information: ")

value_raw = st.number_input("Transaction Value", min_value=0.0, step=0.01, value=5000.0)
aggregate_value_raw = st.number_input(
    "Aggregate Value", min_value=0.0, step=0.01, value=40000.0
)
aggregate_volume_raw = st.number_input(
    "Aggregate Volume", min_value=0.0, step=0.01, value=250.0
)
hour = st.slider("Transaction Hour", 0, 23, 13)
originator_country = st.selectbox("Originator Country", list(country_mapping.keys()))
beneficiary_country = st.selectbox("Beneficiary Country", list(country_mapping.keys()))
currency = st.selectbox("Currency", list(currency_mapping.keys()))
transaction_type = st.selectbox("Transaction Type", list(type_mapping.keys()))

# Prediction
if st.button("Predict Fraud"):
    try:
        # Apply log1p transformation to appropriate features
        value_log = np.log1p(value_raw)
        aggregate_value_log = np.log1p(aggregate_value_raw)
        aggregate_volume_log = np.log1p(aggregate_volume_raw)

        # Encode categorical values
        originator_country_enc = country_mapping.get(originator_country, 0)
        beneficiary_country_enc = country_mapping.get(beneficiary_country, 0)
        currency_enc = currency_mapping.get(currency, 0)
        type_enc = type_mapping.get(transaction_type, 0)

        # Create input array with 11 features
        input_data = np.array(
            [
                value_raw,
                aggregate_value_raw,
                aggregate_volume_raw,
                value_log,
                aggregate_value_log,
                aggregate_volume_log,
                hour,
                originator_country_enc,
                beneficiary_country_enc,
                currency_enc,
                type_enc,
            ]
        ).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0, 1]

        # Display results
        st.markdown("---")
        if prediction[0] == 1:
            st.error(
                f"🚨 **Fraudulent Transaction Detected!**\nFraud Probability: **{probability * 100:.2f}%**"
            )
        else:
            st.success(
                f"✅ **Normal Transaction**\nFraud Probability: **{probability * 100:.2f}%**"
            )
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
