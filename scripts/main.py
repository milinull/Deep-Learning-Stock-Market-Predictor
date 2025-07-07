# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from DL_stock_class import StockPredictor  

app = FastAPI(
    title="API de Previsão de Ações",
    description="API RESTful para previsão de preços com LSTM",
    version="1.0"
)

# Configurações
MAX_PREDICTION_DAYS = 22  # Limite máximo de dias para previsão

class PredictRequest(BaseModel):
    stock: str  # Agora obrigatório, sem valor default
    days_ahead: int

@app.post("/predict")
async def predict_prices(request: PredictRequest):
    try:
        # Acessando os dados validados
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
            #"prediction_days": days,
            #"max_prediction_days": MAX_PREDICTION_DAYS
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
