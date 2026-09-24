import yfinance as yf
import pandas as pd
import numpy as np

def load_and_prep_data(ticker="^GSPC", start_date="2015-01-01", end_date="2023-01-01"):
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Gestione sicura delle colonne per le nuove versioni di yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Mantiene solo il primo livello dei nomi (es. 'Close', 'High')
        df.columns = df.columns.get_level_values(0)
        
    # Rimuove eventuali colonne duplicate generate dalla formattazione di yfinance
    df = df.loc[:, ~df.columns.duplicated()]
        
    # Seleziona 'Adj Close' se esiste, altrimenti ripiega in automatico su 'Close'
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1)) * 100
    df = df.dropna()
    
    df['Realized_Vol'] = df['Log_Return'].rolling(window=21).std() * np.sqrt(252)
    df = df.dropna()
    
    return df
    
    # Calcola la volatilità realizzata (es. deviazione standard mobile a 21 giorni annuata)
    df['Realized_Vol'] = df['Log_Return'].rolling(window=21).std() * np.sqrt(252)
    df = df.dropna()
    
    return df

def split_data(df, train_ratio=0.8):
    # Split temporale rigoroso (no shuffle per serie storiche)
    split_idx = int(len(df) * train_ratio)
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    return train_data, test_data

if __name__ == "__main__":
    df = load_and_prep_data()
    train, test = split_data(df)
    print(f"Dati di training: {len(train)} giorni. Dati di test: {len(test)} giorni.")