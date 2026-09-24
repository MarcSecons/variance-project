import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, sequence_length):
    """
    Crea le sequenze temporali (X) e i target (y) per l'addestramento dell'LSTM.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def preprocess_data(df, target_col='Realized_Volatility', sequence_length=21, test_size=0.2):
    """
    Scala i dati e li divide in train e test mantenendo l'ordine temporale.
    Assumiamo che la Persona A ti passi un DataFrame con la colonna 'Realized_Volatility'.
    """
    # 1. Estrazione dei dati
    values = df[[target_col]].values
    
    # 2. Train-Test Split (Rigorosamente temporale, NON mescolare!)
    split_idx = int(len(values) * (1 - test_size))
    train_data = values[:split_idx]
    test_data = values[split_idx:]
    
    # 3. Scaling (Fitta lo scaler SOLO sui dati di train per evitare data leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # 4. Creazione sequenze
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)
    
    return X_train, y_train, X_test, y_test, scaler, split_idx