from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(sequence_length, n_features=1):
    """
    Costruisce e compila il modello LSTM.
    """
    model = Sequential()
    
    # Layer LSTM principale
    model.add(LSTM(units=50, activation='relu', input_shape=(sequence_length, n_features)))
    
    # Dropout per regolarizzazione (previene l'overfitting)
    model.add(Dropout(0.2))
    
    # Layer di output (1 neurone che prevede la volatilità)
    model.add(Dense(units=1))
    
    # Compilazione del modello
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model