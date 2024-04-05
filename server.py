from flask import Flask, request, jsonify
import json
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def hello():
    request_body = request.json
    sales_data = request_body['sales_data']
    granularidade = request_body['granularidade']  # D, S, M
    periodos = int(request_body['periodos'])  # Quantidade de períodos para previsão
    data_inicio = request_body['data_inicio']  # Data de início para as previsões
    aumento_percentual = float(request_body.get('aumento_percentual', 0))  # Novo parâmetro para aumento percentual

    df_sales = pd.json_normalize(sales_data)
    df_sales['timestamp'] = pd.to_datetime(df_sales['timestamp'])
    freq_map = {'D': 'D', 'S': 'W-MON', 'M': 'M'}  # Ajuste para garantir início na segunda-feira
    df_sales_grouped = df_sales.groupby(['item_id', pd.Grouper(key='timestamp', freq=freq_map[granularidade])])['demand'].sum().reset_index()

    df_previsoes = pd.DataFrame(columns=['mercadoria_id', 'dPrevisao', 'periodo', 'qTotal'])

    start_date = pd.to_datetime(data_inicio)
    if granularidade == 'S':
        start_date = start_date - pd.Timedelta(days=start_date.weekday())

    for produto in df_sales_grouped['item_id'].unique():
        df_produto = df_sales_grouped[df_sales_grouped['item_id'] == produto]
        try:
            #if len(df_produto) > 3:  # Supondo que pelo menos 4 registros são necessários para SARIMA
            model = SARIMAX(df_produto['demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Exemplo básico de SARIMA
            #else:
            #model = ARIMA(df_produto['demand'], order=(1, 1, 1))  # ARIMA como fallback
            model_fit = model.fit()
            date_range = pd.date_range(start=start_date, periods=periodos, freq=freq_map[granularidade])
            forecast = model_fit.forecast(steps=periodos)
            for date, vendas in zip(date_range, forecast):
                vendas_ajustadas = vendas * (1 + aumento_percentual)  # Ajusta as previsões
                formatted_date = date.strftime("%Y-%m-%d")
                df_previsoes.loc[len(df_previsoes)] = [produto, formatted_date, granularidade, round(vendas_ajustadas)]
        except Exception as e:
            print(f"Não foi possível fazer a previsão para o produto {produto}: {e}")

    return jsonify(df_previsoes.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
