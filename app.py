
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import hashlib

# Load the model
with open("./models/casing_quality.pkl", 'rb') as file:
    casing_model = pickle.load(file)

    # with open("D:/Finalyearproject/mushroom-farming-app/models/disease_detection.pkl", 'rb') as file:
    # disease_model = pickle.load(file)

    # with open("D:\Finalyearproject\mushroom-farming-app\models\disease_detection.pkl", 'rb') as file:
    # harvest_model = pickle.load(file)

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to authenticate users
def authenticate(username, password):
    if username in st.session_state['users'] and st.session_state['users'][username] == hash_password(password):
        return True
    return False

# Function to preprocess input data
def preprocess_input(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Initialize session state for users
if 'users' not in st.session_state:
    st.session_state['users'] = {}

# Streamlit app
st.title("Button Mushroom Farming Optimization Using Machine Learning")

# Login page
st.sidebar.title("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    if authenticate(username, password):
        st.sidebar.success("Logged in as {}".format(username))
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
    else:
        st.sidebar.error("Invalid credentials")

# Signup page
st.sidebar.title("Signup")
new_username = st.sidebar.text_input("New Username")
new_password = st.sidebar.text_input("New Password", type="password")

if st.sidebar.button("Signup"):
    if new_username in st.session_state['users']:
        st.sidebar.error("Username already exists")
    else:
        st.session_state['users'][new_username] = hash_password(new_password)
        st.sidebar.success("Account created successfully!")

# Main app content
if 'logged_in' in st.session_state and st.session_state['logged_in']:
    st.write(f"Welcome, {st.session_state['username']}!")

    # Input form for user data
    st.write("Enter the following details for prediction:")
    pH = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
    WHC = st.number_input("Water Holding Capacity (WHC)", min_value=0.0, max_value=100.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'pH': [pH],
        'WHC': [WHC],
        'temperature': [temperature],
        'humidity': [humidity]
    })

    # Preprocess the input data
    input_data_scaled = preprocess_input(input_data)

    # Make a prediction
    if st.button("Predict"):
        prediction = casing_model.predict(input_data_scaled)
        st.write("Prediction:", prediction)
else:
    st.write("Please login to access the features.")
