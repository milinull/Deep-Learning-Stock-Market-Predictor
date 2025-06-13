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

class DlModel:
    def __init__(self):
        self.class_manager = StockData()

        self.x_train, self.y_train = None
        self.x_test, self.y_test = None

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

    def reshape(self):
        # Reshape input to be [samples, time steps, features]
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))