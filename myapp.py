# mushroom_farming_app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set page config
st.set_page_config(page_title="Mushroom Farming Assistant", layout="wide")

# Load models (replace with your actual model loading code)
@st.cache_resource
def load_models():
    try:
        # Casing model
        with open("D:/Finalyearproject/mushroom-farming-app/models/casing_quality_model.pkl", 'rb') as f:
            casing_model = pickle.load(f)
        
        # Disease model
        with open("D:/Finalyearproject/mushroom-farming-app/models/disease_detection_model.pkl", 'rb') as f:
            disease_data = pickle.load(f)
            disease_model = disease_data['model']
            disease_preprocessor = disease_data['preprocessor']
        
        # Harvest model
        with open("D:/Finalyearproject/mushroom-farming-app/models/harvest_prediction_model.pkl", 'rb') as f:
            harvest_data = pickle.load(f)
            harvest_model = harvest_data['model']
            harvest_preprocessor = harvest_data['preprocessor']
        
        return casing_model, disease_model, disease_preprocessor, harvest_model, harvest_preprocessor
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

casing_model, disease_model, disease_preprocessor, harvest_model, harvest_preprocessor = load_models()

# App title and description
st.title("🍄 Mushroom Farming Assistant")
st.markdown("""
This app helps mushroom farmers predict:
- **Casing Quality** (Good/Poor)
- **Disease Risk** (High/Low)
- **Expected Harvest** (in kg)

Adjust the parameters in the sidebar and see the predictions!
""")

# Sidebar for input parameters
st.sidebar.header("Farm Parameters")

# Create tabs for different predictions
tab1, tab2, tab3 = st.tabs(["Casing Quality", "Disease Risk", "Harvest Prediction"])

# Casing Quality Prediction
with tab1:
    st.header("Casing Quality Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ph = st.slider("pH Level", 5.0, 8.5, 6.8, 0.1, key="casing_ph")
        whc = st.slider("Water Holding Capacity (%)", 50, 90, 75, 1, key="casing_whc")
    
    with col2:
        temp = st.slider("Temperature (°C)", 18, 30, 23, 1, key="casing_temp")
        humidity = st.slider("Humidity (%)", 60, 95, 80, 1, key="casing_humidity")
        ec = st.slider("EC (µS/cm)", 100, 1000, 350, 10, key="casing_ec")
    
    if st.button("Predict Casing Quality", key="predict_casing"):
        if casing_model is not None:
            input_data = pd.DataFrame([[ph, whc, temp, humidity, ec]], 
                                    columns=['pH', 'WHC', 'Temperature', 'Humidity', 'EC'])
            prediction = casing_model.predict(input_data)[0]
            proba = casing_model.predict_proba(input_data)[0]
            
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success("✅ Good Casing Quality")
                st.metric("Probability", f"{proba[1]*100:.1f}%")
            else:
                st.warning("⚠️ Poor Casing Quality")
                st.metric("Probability", f"{proba[0]*100:.1f}%")
            
            # Show feature importance if available
            if hasattr(casing_model, 'feature_importances_'):
                st.subheader("Key Factors")
                features = ['pH', 'WHC', 'Temperature', 'Humidity', 'EC']
                importance = casing_model.feature_importances_
                importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
                importance_df = importance_df.sort_values('Importance', ascending=False)
                st.bar_chart(importance_df.set_index('Feature'))
        else:
            st.error("Casing model not loaded properly")

# Disease Risk Prediction
with tab2:
    st.header("Disease Risk Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        d_ph = st.slider("pH Level", 5.0, 8.5, 6.8, 0.1, key="disease_ph")
        d_temp = st.slider("Temperature (°C)", 18, 30, 23, 1, key="disease_temp")
        d_humidity = st.slider("Humidity (%)", 60, 95, 80, 1, key="disease_humidity")
    
    with col2:
        d_light = st.slider("Light Intensity (lux)", 100, 1000, 500, 10, key="disease_light")
        d_co2 = st.slider("CO₂ Level (ppm)", 800, 2000, 1200, 10, key="disease_co2")
        d_ventilation = st.selectbox("Ventilation", ["Low", "Medium", "High"], key="disease_ventilation")
    
    if st.button("Predict Disease Risk", key="predict_disease"):
        if disease_model is not None and disease_preprocessor is not None:
            input_data = pd.DataFrame([[d_ph, d_temp, d_humidity, d_ventilation, d_light, d_co2]], 
                                    columns=['pH', 'Temperature', 'Humidity', 'Ventilation', 'Light_Intensity', 'CO2_Level'])
            
            # Preprocess the input
            processed_data = disease_preprocessor.transform(input_data)
            
            # Make prediction
            prediction = disease_model.predict(processed_data)[0]
            proba = disease_model.predict_proba(processed_data)[0]
            
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("🚨 High Disease Risk")
                st.metric("Probability", f"{proba[1]*100:.1f}%")
                
                st.subheader("Recommendations")
                st.markdown("""
                - Improve ventilation
                - Monitor humidity levels
                - Check for early signs of contamination
                - Consider adjusting temperature
                """)
            else:
                st.success("✅ Low Disease Risk")
                st.metric("Probability", f"{proba[0]*100:.1f}%")
                
                st.subheader("Maintenance Tips")
                st.markdown("""
                - Continue current practices
                - Regular monitoring recommended
                - Maintain proper hygiene
                """)
        else:
            st.error("Disease model not loaded properly")

# Harvest Prediction
with tab3:
    st.header("Harvest Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        h_bags = st.slider("Number of Bags", 50, 500, 200, 10, key="harvest_bags")
        h_days = st.slider("Days Since Casing", 1, 30, 15, 1, key="harvest_days")
    
    with col2:
        h_spawn = st.selectbox("Spawn Type", ["Strain1", "Strain2", "Strain3"], key="harvest_spawn")
        h_casing = st.selectbox("Casing Type", ["Peat", "Coconut", "Pine"], key="harvest_casing")
    
    if st.button("Predict Harvest", key="predict_harvest"):
        if harvest_model is not None and harvest_preprocessor is not None:
            input_data = pd.DataFrame([[h_bags, h_spawn, h_casing, h_days]], 
                                    columns=['Bags', 'Spawn_Type', 'Casing_Type', 'Days_Since_Casing'])
            
            # Preprocess the input
            processed_data = harvest_preprocessor.transform(input_data)
            
            # Make prediction
            prediction = harvest_model.predict(processed_data)[0]
            
            st.subheader("Prediction Result")
            st.metric("Expected Harvest", f"{prediction:.1f} kg")
            
            # Show some context
            if prediction > 300:
                st.success("Excellent expected yield!")
            elif prediction > 200:
                st.info("Good expected yield")
            else:
                st.warning("Below average expected yield")
            
            st.subheader("Optimization Tips")
            st.markdown("""
            - Consider adjusting spawn to casing ratio
            - Monitor environmental conditions
            - Review sterilization procedures
            """)
        else:
            st.error("Harvest model not loaded properly")

# Footer
st.markdown("---")
st.markdown("""
*Note: Predictions are based on machine learning models trained on synthetic data. 
For actual farming decisions, consult with agricultural experts.*
""")