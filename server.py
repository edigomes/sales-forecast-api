from flask import Flask, request, jsonify
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

app = Flask(__name__)

def make_predictions(data, periods, granularidade, start_date):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)


    freq_map = {'D': 'D', 'S': 'W-MON', 'M': 'ME'}
    freq_number = {'D': 30, 'S': 7, 'M': 12}
    df_sales = pd.json_normalize(data)
    df_sales['timestamp'] = pd.to_datetime(df_sales['timestamp'])
    df_sales_grouped = df_sales.groupby(['item_id', pd.Grouper(key='timestamp', freq=freq_map[granularidade])])[
        'demand'].sum().reset_index()

    df_previsoes = pd.DataFrame(columns=['mercadoria_id', 'dPrevisao', 'periodo', 'qTotal'])

    for produto in df_sales_grouped['item_id'].unique():
        df_produto = df_sales_grouped[df_sales_grouped['item_id'] == produto]

        if len(data) < 6:
            model = ARIMA(df_produto['demand'], order=(1, 1, 1))
            model_fit = model.fit()
        else:
            model = SARIMAX(df_produto['demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, freq_number[granularidade]))
            model_fit = model.fit(disp=0)

        date_range = pd.date_range(start=start_date, periods=periods, freq=freq_map[granularidade])

        forecast = model_fit.forecast(steps=periods)
        for date, vendas in zip(date_range, forecast):
            formatted_date = date.strftime("%Y-%m-%d")
            df_previsoes.loc[len(df_previsoes)] = [produto, formatted_date, granularidade, round(vendas)]
    return jsonify(df_previsoes.to_dict(orient='records'))

@app.route('/forecast', methods=['POST'])
def predict():
    data = request.json

    granularidade = data["granularidade"]
    data_inicio = data["data_inicio"]
    periodos = data["periodos"]
    sales_data = data["sales_data"]

    start_date = pd.to_datetime(data_inicio)
    if granularidade == 'S':
        start_date = start_date - timedelta(days=start_date.weekday())

    predictions = make_predictions(sales_data, periodos, granularidade, start_date)

    return predictions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)