import streamlit as st
import pandas as pd
import pickle
import os
import hashlib
from typing import Optional, Tuple


# Set page configuration
st.set_page_config(page_title="Mushroom Farming Assistant", layout="wide")

# --- Model Loading Utilities ---
def verify_model_files() -> list:
    required_files = [
        './models/casing_quality.pkl',
        './models/disease_detection.pkl',
        './models/harvest_prediction.pkl'
    ]
    return [f for f in required_files if not os.path.exists(f)]

@st.cache_resource
def load_models() -> Tuple[Optional[object], Optional[object], Optional[object], Optional[object], Optional[object]]:
    missing_files = verify_model_files()
    if missing_files:
        st.error(f"Missing model files: {', '.join(missing_files)}")
        return None, None, None, None, None

    def safe_load(path: str, requires_preprocessor: bool = False):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if requires_preprocessor:
                if not isinstance(data, dict) or 'model' not in data or 'preprocessor' not in data:
                    return None, None, f"Invalid format in {os.path.basename(path)}"
                return data['model'], data['preprocessor'], None
            return data, None, None
        except Exception as e:
            return None, None, str(e)

    casing_model, _, casing_error = safe_load('./models/casing_quality.pkl')
    disease_model, disease_preprocessor, disease_error = safe_load('./models/disease_detection.pkl', True)
    harvest_model, harvest_preprocessor, harvest_error = safe_load('./models/harvest_prediction.pkl', True)

    for error in [casing_error, disease_error, harvest_error]:
        if error:
            st.error(f"Model loading error: {error}")

    return casing_model, disease_model, disease_preprocessor, harvest_model, harvest_preprocessor

# --- Auth ---
def initialize_session():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'users' not in st.session_state:
        st.session_state.users = {}

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, password):
    if username in st.session_state.users:
        return "Username exists."
    st.session_state.users[username] = hash_password(password)
    return "Registered."

def login_user(username, password):
    if username not in st.session_state.users:
        return "Username not found."
    if st.session_state.users[username] != hash_password(password):
        return "Wrong password."
    st.session_state.logged_in = True
    st.session_state.username = username
    return "Logged in."

# --- Prediction Logic ---
def predict_casing(model, ph, whc, temp, hum, ec):
    if not model:
        return None, "Model missing."
    try:
        df = pd.DataFrame([[ph, whc, temp, hum, ec]], columns=['pH', 'WHC', 'Temperature', 'Humidity', 'EC'])
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else None
        return {'prediction': 'Good' if pred else 'Poor', 'probability': proba}, None
    except Exception as e:
        return None, str(e)

def predict_disease(model, prep, ph, temp, hum, vent, light, co2):
    if not model or not prep:
        return None, "Model or preprocessor missing."
    try:
        df = pd.DataFrame([[ph, temp, hum, vent, light, co2]],
                          columns=['pH', 'Temperature', 'Humidity', 'Ventilation', 'Light_Intensity', 'CO2_Level'])
        proc = prep.transform(df)
        pred = model.predict(proc)[0]
        proba = model.predict_proba(proc)[0][1] if hasattr(model, 'predict_proba') else None
        return {'prediction': 'High' if pred else 'Low', 'probability': proba}, None
    except Exception as e:
        return None, str(e)

def predict_harvest(model, prep, bags, spawn, casing, days):
    if not model or not prep:
        return None, "Model or preprocessor missing."
    try:
        df = pd.DataFrame([[bags, spawn, casing, days]],
                          columns=['Bags', 'Spawn_Type', 'Casing_Type', 'Days_Since_Casing'])
        proc = prep.transform(df)
        pred = model.predict(proc)[0]
        return {'yield': pred}, None
    except Exception as e:
        return None, str(e)

# --- App ---
def main():
    initialize_session()
    casing_model, disease_model, disease_pre, harvest_model, harvest_pre = load_models()

    if not st.session_state.logged_in:
        st.title("Button Mushroom Farming Assistant")
        login, register = st.tabs(["Login", "Register"])

        with login:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    msg = login_user(u, p)
                    if st.session_state.logged_in:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with register:
            with st.form("register_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Register"):
                    st.success(register_user(u, p))
        return

    st.title("Button Mushroom Farming Assistant")
    st.markdown(f"Welcome, **{st.session_state.username}**!")

    tab1, tab2, tab3 = st.tabs(["Casing Quality", "Disease Risk", "Harvest Prediction"])

    with tab1:
        st.header("Casing Quality")
        if not casing_model:
            st.error("Casing model missing.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                ph = st.slider("pH", 5.0, 8.5, 6.8, 0.1)
                whc = st.slider("WHC (%)", 50, 90, 75, 1)
            with c2:
                temp = st.slider("Temp (°C)", 18, 30, 23)
                hum = st.slider("Humidity (%)", 60, 95, 80)
                ec = st.slider("EC", 100, 1000, 350, 10)
            if st.button("Predict Casing"):
                res, err = predict_casing(casing_model, ph, whc, temp, hum, ec)
                if err:
                    st.error(err)
                else:
                    st.success(f"{res['prediction']} quality")
                    if res['probability'] is not None:
                        st.metric("Probability", f"{res['probability']*100:.1f}%")
        st.area_chart(casing_model)
        st.scatter_chart(casing_model)
        st.bar_chart(casing_model)
        

    with tab2:
        st.header("Disease Risk")
        if not disease_model or not disease_pre:
            st.error("Disease model/preprocessor missing.")
        else:
            d1, d2 = st.columns(2)
            with d1:
                ph = st.slider("pH", 5.0, 8.5, 6.8, 0.1, key="dph")
                temp = st.slider("Temp", 18, 30, 23, key="dtemp")
                hum = st.slider("Humidity", 60, 95, 80, key="dhum")
            with d2:
                light = st.slider("Light (lux)", 100, 1000, 500, key="dlight")
                co2 = st.slider("CO2 (ppm)", 800, 2000, 1200, key="dco2")
                vent = st.selectbox("Ventilation", ["Low", "Medium", "High"], key="dvent")
            if st.button("Predict Disease"):
                res, err = predict_disease(disease_model, disease_pre, ph, temp, hum, vent, light, co2)
                if err:
                    st.error(err)
                else:
                    color = "error" if res['prediction'] == 'High' else "success"
                    getattr(st, color)(f"Disease Risk: {res['prediction']}")
                    if res['probability'] is not None:
                        st.metric("Probability", f"{res['probability']*100:.1f}%")
        st.area_chart(disease_model)
        st.scatter_chart(disease_model)
        st.bar_chart(disease_model)
        

    with tab3:
        st.header("Harvest Prediction")
        if not harvest_model or not harvest_pre:
            st.error("Harvest model/preprocessor missing.")
        else:
            h1, h2 = st.columns(2)
            with h1:
                bags = st.slider("Bags", 50, 500, 200, 10)
                days = st.slider("Days", 1, 30, 15)
            with h2:
                spawn = st.selectbox("Spawn Type", ["Strain1", "Strain2", "Strain3"])
                casing = st.selectbox("Casing Type", ["Peat", "Coconut", "Pine"])
            if st.button("Predict Harvest"):
                res, err = predict_harvest(harvest_model, harvest_pre, bags, spawn, casing, days)
                
                if err:
                    st.error(err)
                else:
                    st.metric("Expected Yield", f"{res['yield']:.1f} kg")
        st.area_chart(harvest_model)
        st.scatter_chart(harvest_model)
        st.bar_chart(harvest_model)

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

if __name__ == "__main__":
    main()
