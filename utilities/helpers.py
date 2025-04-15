import numpy as np
import pandas as pd
import hashlib

def make_hashes(password):
    """Create SHA-256 hash of password"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    """Verify password against stored hash"""
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def generate_mushroom_data(n_samples=7000):
    """Generate synthetic mushroom farming data"""
    # Mushroom types (edible and non-edible)
    types = ['Button', 'Shiitake', 'Oyster', 'Poisonous']
    edible = [True, True, True, False]
    
    # Environmental parameters
    temp = np.random.normal(loc=22, scale=3, size=n_samples)
    humidity = np.random.normal(loc=85, scale=5, size=n_samples)
    co2 = np.random.normal(loc=1200, scale=300, size=n_samples)
    ph = np.random.normal(loc=6.5, scale=0.5, size=n_samples)
    
    # Growth days
    growth_days = np.random.randint(15, 30, size=n_samples)
    
    # Casing quality (1-10 scale)
    casing_quality = np.random.randint(3, 10, size=n_samples)
    
    # Disease probability based on conditions
    disease_prob = 0.3 * ((humidity > 90).astype(int) + 
                         (temp > 25).astype(int) + 
                         (casing_quality < 5).astype(int)) / 3
    has_disease = np.random.binomial(1, disease_prob)
    
    # Yield prediction
    yield_factor = (0.4 * (casing_quality/10) + 
                   0.3 * (1 - disease_prob) + 
                   0.2 * ((temp >= 20) & (temp <= 24)).astype(int) + 
                   0.1 * ((humidity >= 80) & (humidity <= 90)).astype(int))
    yield_kg = np.round(yield_factor * np.random.uniform(5, 20, size=n_samples), 2)
    
    # Harvest readiness
    harvest_ready = ((growth_days >= 20) & 
                    (growth_days <= 25) & 
                    (yield_kg > 10) & 
                    (has_disease == 0)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'mushroom_type': np.random.choice(types, size=n_samples, p=[0.4, 0.2, 0.2, 0.2]),
        'temperature': temp,
        'humidity': humidity,
        'co2': co2,
        'ph': ph,
        'growth_days': growth_days,
        'casing_quality': casing_quality,
        'has_disease': has_disease,
        'yield_kg': yield_kg,
        'harvest_ready': harvest_ready
    })
    
    # Add edible flag
    type_to_edible = dict(zip(types, edible))
    data['edible'] = data['mushroom_type'].map(type_to_edible)
    
    # Add realistic patterns
    data.loc[(data['mushroom_type'] == 'Button') & (data['temperature'] > 24), 'yield_kg'] *= 0.8
    data.loc[(data['mushroom_type'] == 'Shiitake') & (data['humidity'] < 80), 'has_disease'] = 1
    data.loc[(data['casing_quality'] < 5) & (data['ph'] < 6), 'harvest_ready'] = 0
    
    return data