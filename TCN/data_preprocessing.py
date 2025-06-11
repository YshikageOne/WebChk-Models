import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def load_data(file_path, sample_frac = 0.1, random_state = 42):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    return df.sample(frac=sample_frac, random_state=random_state)

def preprocess_data(df):
    #clean data
    df = df.dropna()
    df.replace([np.inf, -np.inf], np.nan, inplace = True)
    df = df.dropna()

    #separate features and labels
    features = df.drop('label', axis = 1)
    labels = df['label']

    #encode labels
    labelEncoder = LabelEncoder()
    labels_encoded = labelEncoder.fit_transform(labels)

    #normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    #reshaping for the model
    features_reshaped = features_scaled.reshape(features_scaled.shape[0], 1, features_scaled.shape[1])

    return features_reshaped, labels_encoded, labelEncoder, scaler

def splitData (features, labels, test_size = 0.2, val_size = 0.1):
    features_temp, features_test, labels_temp, labels_test = train_test_split(
        features, labels, test_size = test_size, random_state = 42
    )

    val_ratio = val_size / (1 - test_size)

    features_train, features_val, labels_train, labels_val = train_test_split(
        features_temp, labels_temp, test_size = val_ratio, random_state = 42
    )

    return features_train, features_val, features_test, labels_train, labels_val, labels_test


