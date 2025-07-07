import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense             # type: ignore
import yfinance as yf
import pandas as pd
from datetime import timedelta

class StockData:
    def __init__(self, stock='MSFT', period='24mo'):
        self.stock = stock
        self.period = period
        self.raw_data = None  # DataFrame original
        self.scaled_data = None
        self.scaler = None

    def download_and_prepare_data(self):
        """Download e prepara os dados para o modelo"""
        print(f"Baixando dados da {self.stock}...")
        self.raw_data = yf.download([self.stock], period=self.period)
        self.scaled_data, self.scaler = self._load_data_from_df(self.raw_data)
        return self.scaled_data, self.scaler, self.raw_data

    def _load_data_from_df(self, df):
        """Processa o DataFrame e normaliza os dados"""
        if 'Close' not in df.columns:
            raise ValueError("Coluna 'Close' não encontrada no DataFrame.")

        close_prices = df['Close'].dropna().astype(float).values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        return scaled_data, scaler
    
    def create_dataset(self, data, time_step):
        """Cria dataset para treinamento/teste"""
        x, y = [], []
        for i in range(len(data) - time_step - 1):
            x.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(x), np.array(y)
    
    def create_future_dates(self, last_date, days_ahead=22):
        """Cria datas futuras considerando apenas dias úteis"""
        future_dates = []
        current_date = last_date + timedelta(days=1)
        
        while len(future_dates) < days_ahead:
            if current_date.weekday() < 5:  # Segunda a sexta
                future_dates.append(current_date)
            current_date += timedelta(days=1)
        
        return future_dates

class LSTMModel:
    def __init__(self, time_step=22, epochs=100, batch_size=32):
        self.time_step = time_step
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Dados de treino/teste
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def prepare_train_test_data(self, data, stock_data_manager):
        """Prepara dados de treino e teste"""
        train_size = int(len(data) * 0.8)
        train_data = data[0:train_size, :]
        test_data = data[train_size - self.time_step:, :]

        self.x_train, self.y_train = stock_data_manager.create_dataset(train_data, self.time_step)
        self.x_test, self.y_test = stock_data_manager.create_dataset(test_data, self.time_step)

        # Reshape para LSTM
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

    def build_model(self):
        """Constrói o modelo LSTM"""
        print("Construindo modelo LSTM...")
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.time_step, 1)))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self):
        """Treina o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi construído. Chame build_model() primeiro.")
        
        print("Treinando modelo...")
        self.history = self.model.fit(
            self.x_train, self.y_train, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            verbose=1
        )

    def predict_test_data(self, scaler):
        """Faz previsões no conjunto de teste"""
        print("Fazendo previsões no conjunto de teste...")
        test_predictions = self.model.predict(self.x_test)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_rescaled = scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        return test_predictions, y_test_rescaled

    def predict_future(self, scaler, last_sequence, days_ahead=22):
        """Faz previsões futuras"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            current_input = current_sequence.reshape((1, len(current_sequence), 1))
            next_pred = self.model.predict(current_input, verbose=0)
            predictions.append(next_pred[0, 0])
            current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def save_model(self, filename='lstm_model.h5'):
        """Salva o modelo treinado"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda.")
        
        base_dir = Path(__file__).resolve().parent.parent  # sobe um nível acima de scripts/
        model_dir = base_dir / 'model'
        model_dir.mkdir(exist_ok=True)
        
        save_path = model_dir / filename
        self.model.save(str(save_path))
        print(f"Modelo salvo em {save_path}")

'''    def load_model(self, filename='lstm_model.h5'):
        """Carrega um modelo salvo"""
        base_dir = Path(__file__).resolve().parent.parent  # sobe um nível
        model_path = base_dir / 'model' / filename

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

        self.model = load_model(str(model_path))
        print(f"Modelo {model_path.name} carregado com sucesso.")
'''

class StockPredictor:
    def __init__(self, stock='MSFT', period='24mo', forecast_days=22):
        self.stock_data = StockData(stock, period)
        self.lstm_model = LSTMModel()
        self.forecast_days = forecast_days

        # Dados processados
        self.scaled_data = None
        self.scaler = None
        self.raw_data = None

    def run_prediction(self):
        """Executa todo o pipeline de previsão"""
        # 1. Preparar dados
        self.scaled_data, self.scaler, self.raw_data = self.stock_data.download_and_prepare_data()

        # 2. Preparar dados de treino/teste
        self.lstm_model.prepare_train_test_data(self.scaled_data, self.stock_data)

        # 3. Construir e treinar modelo
        self.lstm_model.build_model()
        self.lstm_model.train_model()

        # 4. Salvar o modelo treinado
        self.lstm_model.save_model('lstm_model.h5')

        # 5. Fazer previsões no teste
        test_predictions, y_test = self.lstm_model.predict_test_data(self.scaler)

        # 6. Fazer previsões futuras
        last_sequence = self.scaled_data[-self.lstm_model.time_step:].flatten()
        future_predictions = self.lstm_model.predict_future(
            self.scaler, last_sequence, self.forecast_days
        )

        # 7. Criar DataFrame com previsões futuras
        last_date = self.raw_data.index[-1].date()
        future_dates = self.stock_data.create_future_dates(last_date, self.forecast_days)

        future_df = pd.DataFrame({
            'Data': future_dates,
            'Previsao_Close': future_predictions
        })

        #print(f"\nPrevisões para os próximos {self.forecast_days} dias úteis:")
        #print(future_df.to_string(index=False))

        return future_df, test_predictions, y_test

# Uso da classe
if __name__ == "__main__":
    predictor = StockPredictor(stock='MSFT', period='24mo', forecast_days=22)
    future_predictions, test_predictions, y_test = predictor.run_prediction()