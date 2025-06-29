from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import numpy as np
from .DL_stock_class import StockPredictor  

app = FastAPI(
    title="API de Previsão de Ações",
    description="API RESTful para previsão de preços com LSTM",
    version="1.0"
)

# Configurações
MAX_PREDICTION_DAYS = 30  # Limite máximo de dias para previsão
DEFAULT_STOCK = 'MSFT'    # Ação padrão se nenhuma for fornecida

class PredictRequest(BaseModel):
    stock: Optional[str] = DEFAULT_STOCK  # Nome da ação (opcional)
    days_ahead: int                       # Número de dias para prever

@app.post("/predict")
async def predict_prices(request: PredictRequest):
    try:
        # 1. Validação dos dados de entrada
        if request.days_ahead <= 0 or request.days_ahead > MAX_PREDICTION_DAYS:
            raise HTTPException(
                status_code=400,
                detail=f"O número de dias para previsão deve estar entre 1 e {MAX_PREDICTION_DAYS}"
            )
        
        # 2. Inicializa o predictor com a ação desejada
        predictor = StockPredictor(stock=request.stock)
        
        # 3. Executa o pipeline de previsão (isso já baixa os dados e treina o modelo)
        future_df, _, _ = predictor.run_prediction()
        
        # 4. Formata a resposta
        predictions = [round(float(p), 2) for p in future_df['Previsao_Close']][:request.days_ahead]
        dates = [str(d) for d in future_df['Data']][:request.days_ahead]
        return {
            "stock": request.stock,
            "predictions": predictions,
            "dates": dates,
            "prediction_days": request.days_ahead,
            "max_prediction_days": MAX_PREDICTION_DAYS
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))