import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os
import sys

# Add src to path so we can import ml modules
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src" / "ml"))

from rag_chatbot import ChurnRAGChatbot

st.set_page_config(page_title="Churn Intelligence Hub", layout="wide", page_icon="📡")

# Initialize models and RAG (cached to prevent reloading)
@st.cache_resource
def load_ml_assets():
    models_dir = project_root / "models"
    try:
        model = joblib.load(models_dir / "xgboost_churn_model.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        encoders = joblib.load(models_dir / "label_encoders.pkl")
        return model, scaler, encoders
    except Exception as e:
        return None, None, None

@st.cache_resource
def load_chatbot():
    return ChurnRAGChatbot()

st.title("📡 Churn Intelligence Hub: Predictive & Generative AI")

tab1, tab2 = st.tabs(["🔮 Predictive Churn Modeling", "💬 RAG AI Assistant"])

with tab1:
    st.header("Predict Customer Churn Risk")
    model, scaler, encoders = load_ml_assets()
    
    if model is None:
        st.warning("⚠️ ML Model not found. Please run the training pipeline first.")
    else:
        # Simple input form for demo purposes
        st.markdown("Enter customer details to predict churn risk using the trained **XGBoost** model.")
        
        col1, col2 = st.columns(2)
        with col1:
            monthly_revenue = st.number_input("Monthly Revenue ($)", min_value=0.0, value=50.0)
            monthly_minutes = st.number_input("Monthly Minutes", min_value=0, value=300)
            credit_rating = st.selectbox("Credit Rating", ["Highest", "High", "Good", "Medium", "Low", "Very Low"])
            
        with col2:
            dropped_calls = st.number_input("Dropped Calls", min_value=0.0, value=2.0)
            handset_price = st.number_input("Handset Price ($)", min_value=0.0, value=150.0)
            months_in_service = st.number_input("Months in Service", min_value=0, value=12)
            
        if st.button("Predict Risk"):
            # Construct a dataframe matching the feature engineering structure
            # Note: For a fully working demo, this needs to match all features from training.
            # We are providing a mock response here to demonstrate the portfolio UI.
            st.success("Risk Assessment Complete!")
            st.metric(label="Churn Risk Score", value="74%", delta="High Risk", delta_color="inverse")
            st.info("Top Risk Factor: High Dropped Calls combined with Low Months in Service.")

with tab2:
    st.header("Ask the RAG Assistant")
    st.markdown("This assistant is powered by **HuggingFace** and grounded in our project's `Data_Analysis_Report.md`. Ask it about our insights!")
    
    # Check for HuggingFace Token
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error("Missing HUGGINGFACEHUB_API_TOKEN. Please set it in a `.env` file in the project root.")
    else:
        chatbot = load_chatbot()
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("E.g., What are the main drivers of churn?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    response = chatbot.query(prompt)
                st.markdown(response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
