import streamlit as st

st.set_page_config(
    page_title="Streamlit ML Workshop",
    layout="wide"
)

st.title("🧪 Streamlit Machine Learning Workshop")

st.markdown("""
Welcome to the **Streamlit ML Workshop**!  
In this workshop, you'll learn how to build an **end-to-end machine learning application**
using Streamlit — from exploration to deployment.
""")

st.header("📚 What This Workshop Covers")

st.markdown("""
### 1️⃣ Exploratory Data Analysis (EDA)
- Visualize datasets
- Identify trends and patterns
- Add written insights directly into the app

### 2️⃣ Model Creation & Evaluation
- Train and fine-tune a machine learning model
- Visualize performance metrics
- Compare model results

### 3️⃣ Model Deployment
- Simulate deploying a trained model
- Accept user inputs
- Generate predictions in real-time
""")

st.info("👉 Use the sidebar to navigate through each stage of the ML workflow.")
