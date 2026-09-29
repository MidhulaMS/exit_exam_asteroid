import pandas as pd
import numpy as np
import pickle
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def main():
    print("Loading dataset...")
    filepath = r"C:\Users\LAB-USER-01\Downloads\asteroid\dataset.csv"
    
    # Load just a sample to be fast
    df = pd.read_csv(filepath, low_memory=False)
    
    # Basic cleaning based on notebook
    df = df.drop(columns=['pdes', 'name', 'prefix', 'albedo', 'diameter_sigma', 
                          'ma', 'epoch', 'om', 'w', 'epoch_cal', 'epoch_mjd', 
                          'id', 'full_name', 'orbit_id'], errors='ignore')
    
    if 'neo' in df.columns:
        df['neo'] = df['neo'].fillna(df['neo'].mode()[0]).map({'Y': 1, 'N': 0})
    if 'pha' in df.columns:
        df['pha'] = df['pha'].fillna(df['pha'].mode()[0]).map({'Y': 1, 'N': 0})
        
    if 'equinox' in df.columns:
        df['equinox'] = df['equinox'].map({'J2000': 1})
        df['equinox'] = df['equinox'].fillna(1)

    df = df.dropna(subset=['diameter'])
    
    # Take a small sample to train quickly
    df = df.sample(n=min(5000, len(df)), random_state=42)
    
    # Drop rows with NaN
    df = df.dropna()
    
    # Categorical encoding for class
    if 'class' in df.columns:
        df = pd.get_dummies(df, columns=['class'], drop_first=True)
        
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=['diameter'])
    y = df['diameter']
    
    # Save feature names
    features = list(X.columns)
    with open('features.json', 'w') as f:
        json.dump(features, f)
        
    # Save median values for default inputs
    medians = X.median().to_dict()
    with open('medians.json', 'w') as f:
        json.dump(medians, f)
        
    print(f"Features count: {len(features)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Training model...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, verbose=0)
    
    model.save('model.h5')
    print("Model and scaler saved.")

if __name__ == "__main__":
    main()
