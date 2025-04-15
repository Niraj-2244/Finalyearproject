import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class MushroomModel:
    def __init__(self):
        self.models = {
            'edibility': None,
            'disease': None,
            'harvest': None,
            'yield': None
        }
        self.le_type = LabelEncoder()
        self.le_material = LabelEncoder()
        
    def generate_data(self, num_samples=1000):
        """Generate synthetic mushroom farming data"""
        np.random.seed(42)
        
        mushroom_types = {
            'Button': {'edible': True, 'temp_range': (18, 24), 'humidity_range': (80, 90), 'growth_days': (18, 25)},
            'Shiitake': {'edible': True, 'temp_range': (12, 18), 'humidity_range': (75, 85), 'growth_days': (20, 28)},
            'Oyster': {'edible': True, 'temp_range': (20, 26), 'humidity_range': (85, 95), 'growth_days': (14, 21)},
            'Poisonous': {'edible': False, 'temp_range': (10, 30), 'humidity_range': (60, 100), 'growth_days': (15, 30)}
        }
        
        data = {
            'mushroom_type': np.random.choice(list(mushroom_types.keys()), size=num_samples, p=[0.4, 0.2, 0.2, 0.2]),
            'temperature': np.zeros(num_samples),
            'humidity': np.zeros(num_samples),
            'co2_level': np.zeros(num_samples),
            'ph_level': np.zeros(num_samples),
            'growth_days': np.zeros(num_samples),
            'casing_material': np.random.choice(['Peat Moss', 'Coir', 'Vermiculite', 'Straw'], size=num_samples),
            'casing_quality': np.random.randint(3, 11, size=num_samples),
            'yield_kg': np.zeros(num_samples),
            'harvest_ready': np.zeros(num_samples),
            'disease_present': np.zeros(num_samples),
            'disease_type': ['None'] * num_samples,
            'edible': np.zeros(num_samples)
        }
        
        for i in range(num_samples):
            m_type = data['mushroom_type'][i]
            props = mushroom_types[m_type]
            
            data['edible'][i] = props['edible']
            data['temperature'][i] = np.random.normal(loc=np.mean(props['temp_range']), scale=2)
            data['humidity'][i] = np.random.normal(loc=np.mean(props['humidity_range']), scale=5)
            data['co2_level'][i] = np.random.normal(loc=1200, scale=200)
            data['ph_level'][i] = np.random.normal(loc=6.5, scale=0.5)
            data['growth_days'][i] = np.random.randint(props['growth_days'][0], props['growth_days'][1]+1)
            
            # Calculate yield
            yield_factor = (
                0.4 * (data['casing_quality'][i]/10) + 
                0.3 * (0.8 if props['temp_range'][0] <= data['temperature'][i] <= props['temp_range'][1] else 0.5) +
                0.2 * (0.9 if props['humidity_range'][0] <= data['humidity'][i] <= props['humidity_range'][1] else 0.6) +
                0.1 * np.random.uniform(0.7, 1.0)
            )
            data['yield_kg'][i] = round(yield_factor * np.random.uniform(5, 20), 2)
            
            # Disease probability
            disease_risk = (
                0.4 * (1 - data['casing_quality'][i]/10) +
                0.3 * (0 if props['temp_range'][0] <= data['temperature'][i] <= props['temp_range'][1] else 0.5) +
                0.3 * (0 if props['humidity_range'][0] <= data['humidity'][i] <= props['humidity_range'][1] else 0.5)
            )
            if np.random.random() < disease_risk:
                data['disease_present'][i] = 1
                data['disease_type'][i] = np.random.choice(
                    ['Bacterial Blotch', 'Green Mold', 'Cobweb Mold', 'Dry Bubble'],
                    p=[0.5, 0.3, 0.15, 0.05]
                )
            
            # Harvest readiness
            data['harvest_ready'][i] = int(
                (props['growth_days'][0] <= data['growth_days'][i] <= props['growth_days'][1]) and
                (data['yield_kg'][i] > 10) and
                (data['disease_present'][i] == 0))
        
        return pd.DataFrame(data)
    
    def prepare_data(self, data):
        """Prepare data for modeling"""
        data = data.copy()
        data['mushroom_type_encoded'] = self.le_type.fit_transform(data['mushroom_type'])
        data['casing_material_encoded'] = self.le_material.fit_transform(data['casing_material'])
        return data
    
    def get_features(self):
        """Get feature columns for modeling"""
        return [
            'temperature', 'humidity', 'co2_level', 'ph_level',
            'growth_days', 'casing_quality',
            'mushroom_type_encoded', 'casing_material_encoded'
        ]
    
    def train_models(self, data):
        """Train all ML models"""
        data = self.prepare_data(data)
        features = self.get_features()
        X = data[features]
        
        # Edibility model
        y_edible = data['edible']
        self.models['edibility'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['edibility'].fit(X, y_edible)
        
        # Disease model
        y_disease = data['disease_present']
        self.models['disease'] = SVC(kernel='rbf', probability=True, random_state=42)
        self.models['disease'].fit(X, y_disease)
        
        # Harvest model
        y_harvest = data['harvest_ready']
        self.models['harvest'] = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.models['harvest'].fit(X, y_harvest)
        
        # Yield model
        y_yield = data['yield_kg']
        self.models['yield'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['yield'].fit(X, y_yield)
        
        return self.models
    
    def save_models(self, path='models'):
        """Save trained models to disk"""
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(path, f'{name}_model.joblib'))
        joblib.dump(self.le_type, os.path.join(path, 'label_encoder_type.joblib'))
        joblib.dump(self.le_material, os.path.join(path, 'label_encoder_material.joblib'))
    
    def load_models(self, path='models'):
        """Load trained models from disk"""
        self.models['edibility'] = joblib.load(os.path.join(path, 'edibility_model.joblib'))
        self.models['disease'] = joblib.load(os.path.join(path, 'disease_model.joblib'))
        self.models['harvest'] = joblib.load(os.path.join(path, 'harvest_model.joblib'))
        self.models['yield'] = joblib.load(os.path.join(path, 'yield_model.joblib'))
        self.le_type = joblib.load(os.path.join(path, 'label_encoder_type.joblib'))
        self.le_material = joblib.load(os.path.join(path, 'label_encoder_material.joblib'))
        return self.models
    
    def predict_edibility(self, input_data):
        """Predict if mushrooms are edible"""
        return self.models['edibility'].predict(input_data), self.models['edibility'].predict_proba(input_data)
    
    def predict_disease(self, input_data):
        """Predict disease probability"""
        return self.models['disease'].predict_proba(input_data)[0][1]
    
    def predict_harvest(self, input_data):
        """Predict harvest readiness"""
        return self.models['harvest'].predict(input_data), self.models['harvest'].predict_proba(input_data)
    
    def predict_yield(self, input_data):
        """Predict yield in kg"""
        return self.models['yield'].predict(input_data)[0]