# 📈 API de Previsão de Ações

API RESTful para previsão de preços de ações utilizando redes LSTM.

---

## 🚀 Como executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```
2. Execute a API:
```bash
uvicorn scripts.main:app --reload
```
3. Acesse a documentação interativa:
http://localhost:8000/docs

## 🛣️ Endpoints

`POST /predict`
Realiza a previsão dos preços de fechamento de uma ação para os próximos dias úteis.

### Request Body
```bash
{
  "stock": "MSFT",
  "days_ahead": 10
}
```

- `stock` (string, obrigatório): Código da ação (ex: `"MSFT"`, `"AAPL"`, `"PETR4.SA"`).
- `days_ahead` (int, obrigatório): Número de dias úteis para prever (1 a 30).

### Exemplo de resposta
```bash
{
  "stock": "MSFT",
  "predictions": [495.86, 495.12, ...],
  "dates": ["2025-06-30", "2025-07-01", ...],
  "prediction_days": 10,
  "max_prediction_days": 30
}
```

- `predictions`: Lista dos valores previstos para fechamento.
- `dates`: Datas correspondentes às previsões.
- `prediction_days`: Dias solicitados na previsão.
- `max_prediction_days`: Limite máximo permitido (30).

### Códigos de resposta
- `200 OK`: Previsão realizada com sucesso.
- `400 Bad Request`: Parâmetros inválidos (ex: dias fora do limite).
- `500 Internal Server Error`: Erro interno ao processar a previsão.

## 🧪 Testando via curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"stock": "MSFT", "days_ahead": 10}'
  ```

## ⚠️ Observações
- O campo `stock` aceita qualquer código de ação suportado pelo Yahoo Finance.
- Se o código informado não existir, a API retornará erro.
- O limite de dias para previsão é de 1 a 30.

## 📚 Documentação automática
Acesse http://localhost:8000/docs para testar a API de forma interativa via Swagger UI.