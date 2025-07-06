# 📈 API de Previsão de Ações com Deep Learning (LSTM)

Este projeto implementa um modelo preditivo utilizando redes neurais LSTM para prever o valor de fechamento de ações da bolsa, com deploy em uma API RESTful desenvolvida em FastAPI.

---

## 📝 Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Requisitos Atendidos](#requisitos-atendidos)
- [Como Executar](#como-executar)
- [Exemplo de Uso da API](#exemplo-de-uso-da-api)
- [Monitoramento e Escalabilidade](#monitoramento-e-escalabilidade)
- [Documentação Automática](#documentação-automática)
- [Extras](#extras)

---

## Sobre o Projeto

Este projeto faz parte do Tech Challenge da Fase 4 e engloba:

- Coleta de dados históricos de ações via Yahoo Finance (`yfinance`)
- Pré-processamento dos dados para uso em redes neurais
- Construção, treinamento e avaliação de um modelo LSTM para séries temporais financeiras
- Salvamento do modelo treinado para inferência
- Deploy do modelo em uma API RESTful (FastAPI)
- Documentação e exemplos de uso

---

## Requisitos Atendidos

- **Coleta e Pré-processamento dos Dados:**  
  Utiliza a biblioteca `yfinance` para baixar dados históricos de qualquer ação suportada pelo Yahoo Finance.  
  Exemplo:
  ```python
  import yfinance as yf
  df = yf.download('MSFT', start='2018-01-01', end='2024-07-20')

- Desenvolvimento do Modelo LSTM:
    - Implementação de rede neural LSTM para previsão de séries temporais.
    - Treinamento e ajuste de hiperparâmetros.
    - Avaliação do modelo com métricas como MAE, RMSE, MAPE.

- Salvamento e Exportação do Modelo:
    - O modelo treinado é salvo em formato .h5 para uso posterior em inferência.

- Deploy do Modelo:
    - API RESTful desenvolvida em FastAPI.
    - Permite ao usuário informar o código da ação e o número de dias para previsão.

- Escalabilidade e Monitoramento:
    - Estrutura pronta para deploy em Docker.
    - Sugestão de uso de ferramentas como Prometheus, Grafana ou New Relic para monitoramento (ver seção Monitoramento e Escalabilidade).


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

## Monitoramento e Escalabilidade

- Monitoramento:
    Recomenda-se o uso de ferramentas como Prometheus, Grafana ou New Relic para monitorar tempo de resposta e uso de recursos da API.

- Escalabilidade:
    O projeto pode ser facilmente containerizado com Docker para deploy em nuvem ou clusters.

## 📚 Documentação automática
Acesse http://localhost:8000/docs para testar a API de forma interativa via Swagger UI.