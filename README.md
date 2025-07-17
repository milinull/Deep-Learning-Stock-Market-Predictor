# 📈 Stock Price Prediction with LSTM

> **MLOps Pipeline para Previsão de Preços de Ações usando Deep Learning**

Sistema completo de predição de preços de ações utilizando redes neurais LSTM (Long Short-Term Memory) com API RESTful para deployment em produção.

![TensorFlow](https://img.shields.io/badge/DL-TensorFlow-FF6F00)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Python](https://img.shields.io/badge/Language-Python-3776AB)
![LSTM](https://img.shields.io/badge/Model-LSTM-FF6B6B)
![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo%20Finance-720E9E)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-4CAF50)

## 📋 Sumário

- [🚀 Características Principais](#-características-principais)
- [🔧 Tecnologias](#-tecnologias)
- [🏗️ Arquitetura](#️-arquitetura)
- [📦 Instalação](#-instalação)
- [🎯 Como Usar](#-como-usar)
- [📊 Modelo e Dados](#-modelo-e-dados)
- [📈 Monitoramento](#-monitoramento)
- [📚 Documentação](#-documentação)
- [👥 Créditos](#-créditos)

## 🚀 Características Principais

- **Deep Learning**: Modelo LSTM para capturar padrões temporais complexos
- **API RESTful**: Interface FastAPI para integração e consumo
- **Dados Reais**: Integração com Yahoo Finance para dados históricos atualizados
- **MLOps Ready**: Pipeline completo de treinamento, validação e deployment
- **Análise Visual**: Notebooks para visualização de resultados e métricas

## 🔧 Tecnologias

- **Deep Learning**: TensorFlow/Keras, LSTM
- **API**: FastAPI, Uvicorn
- **Dados**: yfinance, pandas, numpy, scikit-learn
- **Visualização**: matplotlib, jupyter

## 🏗️ Arquitetura

```
📁 Deep-Learning-Stock-Market-Predictor/
├── 📁 scripts/
│   ├── 📄 DL_stock_class.py    # Classes principais do modelo LSTM  
│   ├── 📄 main.py              # API FastAPI
│   ├── 📄 graphics.ipynb       # Análise visual e métricas
│   ├── 📄 monitoring.py        # Monitoramento de recursos do sistema
├── 📁 model/                   # Modelos treinados salvos
│   ├── 📄 lstm_model.keras
└── 📄 requirements.txt         # Dependências
```

## 📦 Instalação

1. **Clone o repositório**
```bash
git clone https://github.com/milinull/Deep-Learning-Stock-Market-Predictor.git
cd Deep-Learning-Stock-Market-Predictor
```

2. **Crie ambiente virtual e instale dependências**
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac
```

3. **Instale dependências**
```bash
pip install -r requirements.txt
```

4. **Execute a API**
```bash
python scripts/main.py
```

## 🎯 Como Usar

### API Endpoint

**`POST /predict`** - Realiza previsão de preços para os próximos dias úteis

#### Requisição
```json
{
  "stock": "MSFT",
  "days_ahead": 10
}
```

- `stock`: Código da ação (ex: "AAPL", "MSFT", "PETR4.SA")
- `days_ahead`: Número de dias úteis para prever (1 a 22)

#### Resposta
```json
{
  "stock": "MSFT",
  "predictions": [495.86, 495.12, ...],
  "dates": ["2025-06-30", "2025-07-01", ...]
}
```

### Exemplo com curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"stock": "MSFT", "days_ahead": 10}'
```

## 📊 Modelo e Dados

### Coleta e Pré-processamento
- Integração com Yahoo Finance via `yfinance`
- Normalização automática com `MinMaxScaler` (0-1)
- Criação de sequências temporais para LSTM

```python
from DL_stock_class import StockData

data_manager = StockData('MSFT', period='24mo')
scaled_data, scaler, raw_data = data_manager.download_and_prepare_data()
```

### Arquitetura LSTM
- 2 camadas LSTM (50 unidades) + 2 camadas Dense
- Otimização: Adam optimizer com loss MSE
- Janela temporal: 22 dias, 100 épocas
- Métricas: MAE, MSE, RMSE

```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
```

## 📈 Monitoramento

- **Métricas**: MAE, MSE, RMSE para avaliação de performance
- **Visualização**: Gráficos de comparação e análise de erros
- **Logging**: Rastreamento de treinamento e predições
- **Versionamento**: Modelos salvos em formato .h5

### Preparado para Escalabilidade
- 🐳 Docker para containerização
- 📊 Prometheus/Grafana para monitoramento
- ☁️ Cloud services (AWS/GCP/Azure)
- 🔄 CI/CD pipelines

## 📚 Documentação

Acesse `http://localhost:8000/docs` para testar a API interativamente via Swagger UI.

## 👥 Créditos

Desenvolvido por:
- **Raphael Nakamura** - 💻 [GitHub](https://github.com/milinull) | 💼 [LinkedIn](https://www.linkedin.com/in/raphael-nakamura017/)

- **Lucas Lopes** - 💻 [GitHub](https://github.com/Lopeslucas) | 💼 [LinkedIn](https://www.linkedin.com/in/lucas-lopes-633b04123/)

---