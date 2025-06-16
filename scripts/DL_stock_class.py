import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

class StockData:
    def __init__(self):
        self.stock = 'MSFT'
        self.period = '24mo'
        self.data = None
        self.scaler = None

    def prepare_data(self):
        print("Baixando dados da MSFT...")
        df = yf.download([self.stock], period=self.period)
        self.data, self.scaler = self.load_data_from_df(df)

        return self.data, self.scaler

    def load_data_from_df(self, df):
        # Garante que a coluna 'Close' existe
        if 'Close' not in df.columns:
            raise ValueError("Coluna 'Close' não encontrada no DataFrame.")

        # Converte a coluna 'Close' para float e remove valores nulos
        close_prices = df['Close'].dropna().astype(float).values.reshape(-1, 1)

        # Normaliza os dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        return scaled_data, scaler
    
    def create_dataset(self, data, time_step):
        x, y = [], []
        for i in range(len(data) - time_step - 1):
            x.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(x), np.array(y)
    
    def create_future_dates(self, last_date, days_ahead=22):
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

class DlModel:
    def __init__(self):
        self.class_manager = StockData()
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # Processar o dataframe diretamente
        self.data, self.scaler = self.class_manager.prepare_data()
        self.time_step = 22  # Number of time steps to look back

    def train(self):
        # Create train and test sets
        train_size = int(len(self.data) * 0.8)
        train_data = self.data[0:train_size, :]
        test_data = self.data[train_size - self.time_step:, :]

        self.x_train, self.y_train = self.class_manager.create_dataset(train_data, self.time_step)
        self.x_test, self.y_test = self.class_manager.create_dataset(test_data, self.time_step)

        self.reshape()

    def reshape(self):
        # Reshape input to be [samples, time steps, features]
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

    def build_lstm_model(self):
        # Build LSTM model
        print("Construindo modelo LSTM...")
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

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
