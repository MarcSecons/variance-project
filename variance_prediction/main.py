import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import Persona 1
from data_loader import load_and_prep_data
from garch_model import fit_garch, predict_garch_volatility

# Import Persona 2
from ml_preprocessing import preprocess_data
from ml_model import build_lstm_model

def calculate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"--- Metriche per {model_name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    return rmse, mae

def plot_results(dates, realized_vol, garch_vol, ml_vol):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, realized_vol, label='Volatilità Realizzata (Target)', color='black', alpha=0.5, linewidth=1)
    plt.plot(dates, garch_vol, label='Previsione GARCH(1,1)', color='blue', linewidth=1.5)
    plt.plot(dates, ml_vol, label='Previsione LSTM', color='red', alpha=0.8, linewidth=1.5)
    
    plt.title('Confronto Previsione Volatilità: GARCH vs LSTM')
    plt.xlabel('Data')
    plt.ylabel('Volatilità Annualizzata (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('volatility_comparison_final.png')
    plt.show()

if __name__ == "__main__":
    # Parametri globali
    TICKER = "^GSPC"
    TRAIN_RATIO = 0.8
    SEQ_LENGTH = 21

    print("1. Scaricamento e preparazione dati...")
    df = load_and_prep_data(ticker=TICKER, start_date="2015-01-01", end_date="2023-01-01")
    
    # ---------------------------------------------------------
    # PIPELINE GARCH (Persona 1)
    # ---------------------------------------------------------
    print("\n2. Esecuzione pipeline GARCH...")
    split_idx_garch = int(len(df) * TRAIN_RATIO)
    train_garch = df.iloc[:split_idx_garch]
    test_garch = df.iloc[split_idx_garch:]
    
    garch_res = fit_garch(train_garch['Log_Return'])
    garch_predictions_full = predict_garch_volatility(garch_res, len(test_garch))

    # ---------------------------------------------------------
    # PIPELINE LSTM (Persona 2)
    # ---------------------------------------------------------
    print("\n3. Esecuzione pipeline LSTM...")
    X_train, y_train, X_test, y_test, scaler, split_idx_ml = preprocess_data(
        df, 
        target_col='Realized_Vol', 
        sequence_length=SEQ_LENGTH, 
        test_size=(1 - TRAIN_RATIO)
    )

    model = build_lstm_model(sequence_length=SEQ_LENGTH, n_features=1)
    
    model.fit(
        X_train, y_train,
        epochs=20,          
        batch_size=32,
        validation_split=0.1, 
        verbose=0 # Impostato a 0 per pulizia dell'output
    )

    y_pred_scaled = model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(y_pred_scaled).flatten()
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # ---------------------------------------------------------
    # ALLINEAMENTO E VALUTAZIONE
    # ---------------------------------------------------------
    print("\n4. Valutazione modelli...")
    # Il test set LSTM inizia dopo SEQ_LENGTH giorni[cite: 3]
    # Tagliamo le prime SEQ_LENGTH previsioni del GARCH e le date per allineare i vettori
    garch_predictions_aligned = garch_predictions_full[SEQ_LENGTH:]
    test_dates_aligned = df.index[split_idx_ml + SEQ_LENGTH:]

    # Controllo di sicurezza sulle dimensioni
    assert len(test_dates_aligned) == len(lstm_predictions) == len(garch_predictions_aligned) == len(y_test_original)

    calculate_metrics(y_test_original, garch_predictions_aligned, "GARCH(1,1)")
    print("")
    calculate_metrics(y_test_original, lstm_predictions, "LSTM")

    print("\n5. Generazione grafico comparativo...")
    plot_results(test_dates_aligned, y_test_original, garch_predictions_aligned, lstm_predictions)