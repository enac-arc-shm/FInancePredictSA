import multiprocessing
import base64
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from database_gcp import upload_data, data_reference
from io import BytesIO
from data_model_class import MyFInance, MLFInance
from flask import Flask, redirect, render_template, jsonify, request, url_for
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,GRU

MLFInanceObj = MLFInance()
FInanceObj = MyFInance()
profit = 0
app = Flask(__name__)

@app.route('/')
def index():
    data_analysis = {
        'company':FInanceObj.get_company(),
        'date_start':FInanceObj.get_date_start_train(),
        'date_end':FInanceObj.get_data_end_train(),
        'profit':profit
    }
    hist = MLFInanceObj.get_hist()
    limited_data = hist.head(10)
    table_data = limited_data.reset_index().to_dict(orient='records')
    return render_template('index.html', table_data=table_data, img_base64=MLFInanceObj.get_img_base64(), data=data_analysis)

@app.route('/train_model', methods={'POST'})
def train_model():
    try:
        input_start_date = request.form['start_date']
        input_end_date = request.form['end_date']
        input_company = request.form['company']
        FInanceObj.set_date_start_train(input_start_date)
        FInanceObj.set_data_end_train(input_end_date)
        FInanceObj.set_company(input_company)
        Intelligence_model()
        result = {'status': 'success', 'message': 'Model trained successfully'}
        time.sleep(2)
        return jsonify(result)
        #return redirect(url_for('index'))
    except Exception as e:
        error_result = {'status': 'error', 'message': str(e)}
        return jsonify(error_result), 500  # Return a 500 Internal Server Error response
    
@app.route('/make_prediction', methods={'POST'})
def make_prediction():
    try:
        input_start_date = request.form['start_date']
        input_end_date = request.form['end_date']
        FInanceObj.set_start_prediction(input_start_date)
        FInanceObj.set_end_prediction(input_end_date)
        prediction_visualización()
        return redirect(url_for('index'))
    except Exception as e:
        error_result = {'status': 'error', 'message': str(e)}
        return jsonify(error_result), 500  # Return a 500 Internal Server Error response

def data_recollection():
    try:
        data_reference()
    except Exception as e:
        print(f"[[X] Error, no response from db {e}")
    company = FInanceObj.get_company()
    start = FInanceObj.get_date_start_train()
    end = FInanceObj.get_data_end_train()
    ticker = yf.Ticker(company)
    hist = ticker.history(start=start, end=end)
    MLFInanceObj.set_hist(hist)
    MLFInanceObj.set_ticker(hist)
    return hist, ticker

def Intelligence_model():
    import matplotlib
    matplotlib.use('Agg')  # Set the backend to Agg
    import matplotlib.pyplot as plt

    #Get data in object
    hist, ticker = data_recollection()

    #Preparar los datos
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(hist['Close'].values.reshape(-1,1))

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days,len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x,0])
        y_train.append(scaled_data[x,0])

    x_train,y_train = np.array(x_train),np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    x_train.shape


    #Contruir el modelo
    model = Sequential()

    model.add(GRU(units=50,return_sequences = True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(GRU(units=50,return_sequences = True))
    model.add(Dropout(0.2))
    model.add(GRU(units=50))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train,y_train,epochs=25,batch_size=32)

    #Cargar los datos del test
    MLFInanceObj.set_scaler(scaler)
    MLFInanceObj.set_Model(model)
    prediction_visualización()


def prediction_visualización():
    #Get data in object
    company = FInanceObj.get_company()
    hist, ticker = data_recollection()
    prediction_days = 60
    scaler = MLFInanceObj.get_scaler()
    model = MLFInanceObj.get_Model()
    start_prediction = FInanceObj.get_start_prediction()
    end_prediction = FInanceObj.get_end_prediction()

    background_process = multiprocessing.Process(target=background_task)
    background_process.start()

    hist_test = ticker.history(start = start_prediction, end=end_prediction)
    actual_prices = hist_test["Close"].values

    total_dataset = pd.concat((hist['Close'],hist_test['Close']),axis=0)
    model_inputs = total_dataset[len(total_dataset)-len(hist_test)-prediction_days:].values
    model_inputs = scaler.transform(model_inputs.reshape(-1,1))


    x_test = []

    for x in range(prediction_days,len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plt.plot(actual_prices,color="black",label=f"{company} real prices")
    plt.plot(predicted_prices,color="blue",label=f"{company} predicted prices")
    plt.legend()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.clf()  # Clear the figure
    img.seek(0)
    plt.close()

    # Convert the BytesIO object to a base64-encoded string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    rentability = 1
    for i in range(1,len(actual_prices)):
        if predicted_prices[i] > actual_prices[i-1]:
            rentability*= actual_prices[i]/actual_prices[i-1]
    global profit 
    profit = (f'{(rentability-1)*100}%')

    MLFInanceObj.set_img_base64(img_base64)

def background_task():
    company = FInanceObj.get_company()
    hist, ticker = data_recollection()
    upload_data(data=hist, column_family_id=company)

if __name__ == '__main__':
    data_recollection()
    Intelligence_model()
    app.run(debug=True, extra_files=['./static'], port='6670')
