import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import sqlite3
import os
from models import MushroomModel

# Authentication Functions
def create_usertable():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

def view_all_users():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    conn.close()
    return data

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Page Functions
def home_page():
    st.header("Welcome to Mushroom Farming Optimizer")
    st.write(f"Hello {st.session_state.username}! This application helps optimize mushroom cultivation.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"Total records: {len(st.session_state.data)}")
        st.write("Mushroom types distribution:")
        st.bar_chart(st.session_state.data['mushroom_type'].value_counts())
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Average Yield (kg)", round(st.session_state.data['yield_kg'].mean(), 2))
        st.metric("Disease Rate", f"{st.session_state.data['disease_present'].mean()*100:.1f}%")
        st.metric("Harvest Readiness", f"{st.session_state.data['harvest_ready'].mean()*100:.1f}%")
    
    st.subheader("Sample Data")
    st.dataframe(st.session_state.data.head())

def data_exploration_page():
    st.header("Data Exploration")
    
    tab1, tab2 = st.tabs(["Summary", "Visualizations"])
    
    with tab1:
        st.subheader("Statistical Summary")
        st.write(st.session_state.data.describe())
        
    with tab2:
        st.subheader("Interactive Visualizations")
        x_axis = st.selectbox("X-axis", ['temperature', 'humidity', 'ph_level', 'co2_level'])
        y_axis = st.selectbox("Y-axis", ['yield_kg', 'growth_days', 'casing_quality'])
        st.scatter_chart(st.session_state.data, x=x_axis, y=y_axis, color='mushroom_type')

def edibility_classifier_page():
    st.header("Edibility Classifier")
    
    with st.form("edibility_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = st.slider("Temperature (°C)", 10.0, 30.0, 22.0)
            humidity = st.slider("Humidity (%)", 60.0, 100.0, 85.0)
        with col2:
            co2 = st.slider("CO2 Level (ppm)", 800, 2000, 1200)
            ph = st.slider("pH Level", 5.0, 8.0, 6.5)
        
        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = pd.DataFrame({
                'temperature': [temp],
                'humidity': [humidity],
                'co2_level': [co2],
                'ph_level': [ph],
                'growth_days': [20],  # Default values
                'casing_quality': [7],
                'mushroom_type_encoded': [0],
                'casing_material_encoded': [0]
            })
            
            prediction, proba = st.session_state.models.predict_edibility(input_data)
            
            if prediction[0]:
                st.success(f"🍄 Edible (Probability: {proba[0][1]*100:.1f}%)")
            else:
                st.error(f"☠️ Non-Edible (Probability: {proba[0][0]*100:.1f}%)")

def disease_detector_page():
    st.header("Disease Detector")
    
    with st.form("disease_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = st.slider("Temperature (°C)", 10.0, 30.0, 22.0)
            humidity = st.slider("Humidity (%)", 60.0, 100.0, 85.0)
        with col2:
            co2 = st.slider("CO2 Level (ppm)", 800, 2000, 1200)
            casing_quality = st.slider("Casing Quality (1-10)", 1, 10, 7)
        
        submitted = st.form_submit_button("Assess Disease Risk")
        if submitted:
            input_data = pd.DataFrame({
                'temperature': [temp],
                'humidity': [humidity],
                'co2_level': [co2],
                'ph_level': [6.5],  # Default values
                'growth_days': [20],
                'casing_quality': [casing_quality],
                'mushroom_type_encoded': [0],
                'casing_material_encoded': [0]
            })
            
            risk = st.session_state.models.predict_disease(input_data)
            
            if risk > 0.7:
                st.error(f"🚨 High disease risk ({risk*100:.1f}%)")
            elif risk > 0.3:
                st.warning(f"⚠️ Moderate disease risk ({risk*100:.1f}%)")
            else:
                st.success(f"✅ Low disease risk ({risk*100:.1f}%)")

def harvest_predictor_page():
    st.header("Harvest Predictor")
    
    with st.form("harvest_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = st.slider("Temperature (°C)", 10.0, 30.0, 22.0)
            humidity = st.slider("Humidity (%)", 60.0, 100.0, 85.0)
        with col2:
            growth_days = st.slider("Growth Days", 10, 30, 20)
            yield_kg = st.slider("Current Yield (kg)", 5.0, 30.0, 15.0)
        
        submitted = st.form_submit_button("Predict Harvest Readiness")
        if submitted:
            input_data = pd.DataFrame({
                'temperature': [temp],
                'humidity': [humidity],
                'co2_level': [1200],  # Default values
                'ph_level': [6.5],
                'growth_days': [growth_days],
                'casing_quality': [7],
                'mushroom_type_encoded': [0],
                'casing_material_encoded': [0]
            })
            
            prediction, _ = st.session_state.models.predict_harvest(input_data)
            
            if prediction[0]:
                st.success("✅ Ready to harvest!")
            else:
                st.warning("⏳ Not ready yet")

def yield_forecast_page():
    st.header("Yield Forecast")
    
    with st.form("yield_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = st.slider("Temperature (°C)", 10.0, 30.0, 22.0)
            humidity = st.slider("Humidity (%)", 60.0, 100.0, 85.0)
        with col2:
            casing_quality = st.slider("Casing Quality (1-10)", 1, 10, 7)
            growth_days = st.slider("Growth Days", 10, 30, 20)
        
        submitted = st.form_submit_button("Predict Yield")
        if submitted:
            input_data = pd.DataFrame({
                'temperature': [temp],
                'humidity': [humidity],
                'co2_level': [1200],  # Default values
                'ph_level': [6.5],
                'growth_days': [growth_days],
                'casing_quality': [casing_quality],
                'mushroom_type_encoded': [0],
                'casing_material_encoded': [0]
            })
            
            prediction = st.session_state.models.predict_yield(input_data)
            st.metric("Predicted Yield", f"{prediction:.2f} kg")

def admin_panel():
    st.header("Admin Panel")
    st.write("User management and system configuration")
    
    st.subheader("Registered Users")
    users = view_all_users()
    st.write(pd.DataFrame(users, columns=["Username", "Password Hash"]))
    
    if st.button("Regenerate Sample Data"):
        st.session_state.data = st.session_state.models.generate_data()
        st.session_state.data.to_csv("data/mushroom_data.csv", index=False)
        st.success("Data regenerated!")

def login_page():
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Login":
        st.sidebar.subheader("Login Section")
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        
        if st.sidebar.checkbox("Login"):
            hashed_pwd = make_hashes(password)
            result = login_user(username, hashed_pwd)
            
            if result:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.warning("Incorrect Username/Password")
                
    elif choice == "Sign Up":
        st.sidebar.subheader("Create New Account")
        new_user = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("Password", type='password')
        
        if st.sidebar.button("Sign Up"):
            if new_user and new_password:
                add_userdata(new_user, make_hashes(new_password))
                st.success("Account created!")
            else:
                st.warning("Enter both username and password")

def main_app():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
    
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    
    # Initialize data
    if 'data' not in st.session_state:
        if os.path.exists("data/mushroom_data.csv"):
            st.session_state.data = pd.read_csv("data/mushroom_data.csv")
        else:
            os.makedirs("data", exist_ok=True)
            st.session_state.data = st.session_state.models.generate_data()
            st.session_state.data.to_csv("data/mushroom_data.csv", index=False)
    
    # Initialize models
    if 'models' not in st.session_state:
        st.session_state.models = MushroomModel()
        if os.path.exists("models"):
            try:
                st.session_state.models.load_models()
            except:
                st.session_state.models.train_models(st.session_state.data)
                st.session_state.models.save_models()
        else:
            st.session_state.models.train_models(st.session_state.data)
            st.session_state.models.save_models()
    
    # Navigation
    pages = {
        "Home": home_page,
        "Data Exploration": data_exploration_page,
        "Edibility Classifier": edibility_classifier_page,
        "Disease Detector": disease_detector_page,
        "Harvest Predictor": harvest_predictor_page,
        "Yield Forecast": yield_forecast_page
    }
    
    if st.session_state.username == "admin":
        pages["Admin Panel"] = admin_panel
    
    page = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[page]()

def main():
    st.set_page_config(page_title="Mushroom Farming Optimizer", page_icon="🍄")
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    create_usertable()
    
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == '__main__':
    main()