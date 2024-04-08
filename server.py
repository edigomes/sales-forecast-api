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
    if len(data) < 6:
        model = ARIMA(df['demand'], order=(1, 1, 1))
        model_fit = model.fit()
    else:
        model = SARIMAX(df['demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, freq_number[granularidade]))
        model_fit = model.fit(disp=0)

    future_dates = pd.date_range(start=start_date, periods=periods, freq=freq_map[granularidade])
    future_dates_str = [date.strftime("%Y-%m-%d") for date in future_dates]

    forecast = model_fit.forecast(steps=periods)

    return [{"dPrevisao": date, "mercadoria_id": data[0]["item_id"], "periodo": "D", "qTotal": int(prediction)} for
            date, prediction in zip(future_dates_str, forecast)]

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

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)