# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template
import utils
import train_models as tm
import pandas as pd

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    all_colors = {'SVR_linear': '#FF9EDD',
                  'SVR_poly': '#FFFD7F',
                  'SVR_rbf': '#FFA646',
                  'linear_regression': '#CC2A1E',
                  'random_forests': '#8F0099',
                  'KNN': '#CCAB43',
                  'DT': '#85CC43',
                  'LSTM_model': '#CC7674'}
    all_files = utils.read_all_stock_files('individual_stocks_5yr')
    df = pd.read_csv('GOOG_30_days.csv')
    dates, prices, ml_models_outputs = tm.train_predict_plot('GOOG', df, ['LSTM_model', 'SVR_rbf', 'DT'])

    all_data = []
    all_data.append((prices, "false", "Data", '#000000'))
    for model_output in ml_models_outputs:
        all_data.append(((ml_models_outputs[model_output])[0], "true", model_output, all_colors[model_output]))

    print(all_data)

    return render_template('index.html', labels=dates, all_date=all_data)


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
