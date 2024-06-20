from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

def make_predictions(data, periods, granularidade, start_date, apply_growth_curve, adjustment_factor):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    freq_map = {'D': 'D', 'S': 'W-MON', 'M': 'M'}
    df_sales = pd.json_normalize(data)
    df_sales['timestamp'] = pd.to_datetime(df_sales['timestamp'])
    df_sales_grouped = df_sales.groupby(['item_id', pd.Grouper(key='timestamp', freq=freq_map[granularidade])])['demand'].sum().reset_index()

    df_previsoes = pd.DataFrame(columns=['mercadoria_id', 'dPrevisao', 'periodo', 'qTotal'])

    for produto in df_sales_grouped['item_id'].unique():
        df_produto = df_sales_grouped[df_sales_grouped['item_id'] == produto]
        df_produto.set_index('timestamp', inplace=True)

        date_range = pd.date_range(start=start_date, periods=periods, freq=freq_map[granularidade])
        
        for i, date in enumerate(date_range):
            formatted_date = date.strftime("%Y-%m-%d")
            year_ago_value = df_produto[df_produto.index == (date - pd.DateOffset(years=1))]['demand'].values
            if len(year_ago_value) > 0:
                vendas = year_ago_value[0]
            else:
                vendas = df_produto['demand'].mean()  # Média dos meses anteriores se não houver dados do ano passado
            
            if apply_growth_curve:
                growth_factor = 1 + 0.1 * (i / periods) if i < periods / 2 else 1 - 0.1 * ((i - periods / 2) / periods)
                vendas = vendas * growth_factor
            
            if adjustment_factor is not None:
                vendas = vendas * (1 + adjustment_factor / 100)
            
            df_previsoes.loc[len(df_previsoes)] = [produto, formatted_date, granularidade, max(0, round(vendas))]

    return jsonify(df_previsoes.to_dict(orient='records'))

@app.route('/forecast', methods=['POST'])
def predict():
    data = request.json

    granularidade = data["granularidade"]
    data_inicio = data["data_inicio"]
    periodos = data["periodos"]
    sales_data = data["sales_data"]
    apply_growth_curve = data.get("apply_growth_curve", False)
    adjustment_factor = data.get("adjustment_factor", None)

    start_date = pd.to_datetime(data_inicio)
    if granularidade == 'S':
        start_date = start_date - timedelta(days=start_date.weekday())

    predictions = make_predictions(sales_data, periodos, granularidade, start_date, apply_growth_curve, adjustment_factor)

    return predictions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)