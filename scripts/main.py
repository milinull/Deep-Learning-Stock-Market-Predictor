from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from DL_stock_class import DlModel
import numpy as np
import datetime

app = FastAPI(
    title="API de Previsão de Ações",
    description="API RESTful para previsão de preços futuros com LSTM",
    version="1.0"
)

# Instanciando o modelo
modelo_dl = DlModel()
modelo_dl.train()
modelo_dl.build_lstm_model()
modelo_dl.model.fit(modelo_dl.x_train, modelo_dl.y_train, batch_size=1, epochs=1)

class PredictRequest(BaseModel):
    last_sequence: List[float]
    days_ahead: int

@app.post("/predict")
def predict_prices(request: PredictRequest):
    try:
        if len(request.last_sequence) != modelo_dl.time_step:
            raise HTTPException(status_code=400, detail=f"Sequência deve ter {modelo_dl.time_step} valores")

        # Normaliza a sequência de entrada
        scaled_sequence = modelo_dl.scaler.transform(np.array(request.last_sequence).reshape(-1, 1)).flatten()

        # Faz a previsão
        predictions = DlModel.predict_future(
            modelo_dl.model,
            modelo_dl.scaler,
            last_sequence=np.array(scaled_sequence),
            days_ahead=request.days_ahead
        )

        # Gera datas futuras
        last_date = datetime.date.today()
        future_dates = modelo_dl.class_manager.create_future_dates(last_date, request.days_ahead)

        return {
            "predictions": predictions.tolist(),
            "dates": [d.strftime('%Y-%m-%d') for d in future_dates]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
