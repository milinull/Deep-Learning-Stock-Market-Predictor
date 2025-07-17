# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from DL_stock_class import StockPredictor 

"""API RESTful para previsão de preços de ações com LSTM.

Este serviço usa um modelo LSTM para prever os preços de fechamento de ações 
com base em dados históricos obtidos via Yahoo Finance. A API é construída com FastAPI 
e oferece um endpoint principal para requisições de previsão.
"""

app = FastAPI(
    title="API de Previsão de Ações",
    description="API RESTful para previsão de preços com LSTM",
    version="1.0"
)

# Configurações
MAX_PREDICTION_DAYS = 22  # Define o limite máximo de dias que podem ser previstos pela API

class PredictRequest(BaseModel):
    """Modelo de entrada para a requisição de previsão.
    Contém o código da ação e o número de dias úteis futuros que se deseja prever.
    """

    stock: str  # Símbolo da ação (ex: 'AAPL', 'MSFT')
    days_ahead: int  # Número de dias úteis a serem previstos


@app.post("/predict")
async def predict_prices(request: PredictRequest):
    """Endpoint principal para prever preços de ações.

    Recebe o código da ação e a quantidade de dias a prever.
    Retorna uma lista com as datas futuras e os valores previstos para cada dia.

    Em caso de erro de validação ou falha na execução, retorna um erro HTTP apropriado.
    """
    try:
        stock = request.stock
        days = request.days_ahead

        if days <= 0 or days > MAX_PREDICTION_DAYS:
            raise HTTPException(
                status_code=400,
                detail=f"O número de dias para previsão deve estar entre 1 e {MAX_PREDICTION_DAYS}"
            )

        predictor = StockPredictor(stock=stock, forecast_days=days)
        future_df, _, _ = predictor.run_prediction()

        predictions = [round(float(p), 2) for p in future_df['Previsao_Close']]
        dates = [str(d) for d in future_df['Data']]

        return {
            "stock": stock,
            "predictions": predictions,
            "dates": dates,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
