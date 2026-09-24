from arch import arch_model
import numpy as np

def fit_garch(train_returns):
    # Inizializza e adatta il modello GARCH(1,1)
    # rescale=False perché i log return sono già moltiplicati per 100 in data_loader
    model = arch_model(train_returns, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
    res = model.fit(disp='off')
    return res

def predict_garch_volatility(res, test_len):
    # Previsione della varianza
    forecasts = res.forecast(horizon=test_len, reindex=False)
    
    # Estrazione della varianza prevista e conversione in volatilità annualizzata
    # forecast.variance.values[-1] contiene le previsioni per l'orizzonte richiesto
    pred_var = forecasts.variance.values[-1]
    pred_vol = np.sqrt(pred_var) * np.sqrt(252)
    
    return pred_vol