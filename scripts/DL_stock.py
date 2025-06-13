import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

def load_data_from_df(df):
    # Garante que a coluna 'Close' existe
    if 'Close' not in df.columns:
        raise ValueError("Coluna 'Close' não encontrada no DataFrame.")

    # Converte a coluna 'Close' para float e remove valores nulos
    close_prices = df['Close'].dropna().astype(float).values.reshape(-1, 1)

    # Normaliza os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    return scaled_data, scaler

def create_dataset(data, time_step):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

def predict_future(model, scaler, last_sequence, days_ahead=22):
    """
    Função para fazer previsões futuras usando o modelo treinado
    
    Args:
        model: Modelo LSTM treinado
        scaler: Scaler usado para normalização
        last_sequence: Últimos 'time_step' valores para iniciar a previsão
        days_ahead: Número de dias para prever (default: 22 dias úteis = ~1 mês)
    
    Returns:
        Array com as previsões desnormalizadas
    """

    # ENTRADA: Números (preços dos últimos 22 dias)
    # SAÍDA: Números (previsões de preços futuros)

    predictions = []    # Lista de VALORES [462.27, 457.33, 451.00, 444.18, ...]
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Reshape para o formato esperado pelo modelo
        current_input = current_sequence.reshape((1, len(current_sequence), 1))
        
        # Fazer previsão
        next_pred = model.predict(current_input, verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Atualizar a sequência: remove o primeiro valor e adiciona a previsão
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # Desnormalizar as previsões
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()

def create_future_dates(last_date, days_ahead=22):
    """
    Cria datas futuras considerando apenas dias úteis
    """

    # ENTRADA: Data (última data conhecida)
    # SAÍDA: Lista de datas (próximos dias úteis)


    future_dates = []   # Lista de DATAS [2025-06-12, 2025-06-13, 2025-06-16, ...]
    current_date = last_date + timedelta(days=1)
    
    while len(future_dates) < days_ahead:
        # Adiciona apenas dias úteis (segunda a sexta)
        if current_date.weekday() < 5:
            future_dates.append(current_date)
        current_date += timedelta(days=1)
    
    return future_dates

# Define hyperparameters
time_step = 22  # Number of time steps to look back
epochs = 100
batch_size = 32

forecast_days = 22  # Aproximadamente 1 mês de dias úteis

# Dataframe dos dados
print("Baixando dados da MSFT...")
dados = yf.download(['MSFT'], period='24mo')

# Processar o dataframe diretamente
data, scaler = load_data_from_df(dados)

# Create train and test sets
train_size = int(len(data) * 0.8)
train_data = data[0:train_size, :]
test_data = data[train_size - time_step:, :]

x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Build LSTM model
print("Construindo modelo LSTM...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("Treinando modelo...")
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Make predictions on test data
print("Fazendo previsões no conjunto de teste...")
test_predictions = model.predict(x_test)
test_predictions = scaler.inverse_transform(test_predictions)

# Desnormalizar para vizualizar no gráfico
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Obter os últimos time_step de valores dos dados originais para previsão futura
last_sequence = data[-time_step:]  # Últimos 22 valores normalizados

# Fazer previsões futuras
print(f"Gerando previsões para os próximos {forecast_days} dias úteis...")
future_predictions = predict_future(model, scaler, last_sequence.flatten(), forecast_days)

# Criar datas futuras
last_date = dados.index[-1].date()
future_dates = create_future_dates(last_date, forecast_days)

# Criar DataFrame com as previsões futuras
future_df = pd.DataFrame({
    'Data': future_dates,
    'Previsao_Close': future_predictions
})

print("\nPrevisões para o próximo mês:")
print(future_df.to_string(index=False))