class MyFInance:
    def __init__(self, company=None, date_start_train=None, data_end_train=None, prediction_layers=None, start_prediction=None, end_prediction=None):
        self._company = company if company is not None else 'AMZN'
        self._date_start_train = date_start_train if data_end_train is not None else '2012-1-1'
        self._data_end_train = data_end_train if data_end_train is not None else '2020-1-1'
        self._prediction_layers = prediction_layers if prediction_layers is not None else 25
        self._start_prediction = start_prediction if start_prediction is not None else '2023-6-1'
        self._end_prediction = end_prediction if end_prediction is not None else '2023-11-27'

    def get_company(self):
        return self._company

    def set_company(self, value):
        self._company = value

    def get_date_start_train(self):
        return self._date_start_train

    def set_date_start_train(self, value):
        self._date_start_train = value

    def get_data_end_train(self):
        return self._data_end_train

    def set_data_end_train(self, value):
        self._data_end_train = value

    def get_prediction_layers(self):
        return self._prediction_layers

    def set_prediction_layers(self, value):
        self._prediction_layers = value

    def get_start_prediction(self):
        return self._start_prediction

    def set_start_prediction(self, value):
        self._start_prediction = value

    def get_end_prediction(self):
        return self._end_prediction

    def set_end_prediction(self, value):
        self._end_prediction = value


class MLFInance:
    def __init__(self, img_base64=None, hist=None, ticker=None, scaler=None, Model=None):
        self._img_base64 = img_base64 if img_base64 is not None else 1
        self._hist = hist if hist is not None else 1
        self._ticker = ticker if ticker is not None else 1
        self._scaler = scaler if scaler is not None else 1
        self._Model = Model if Model is not None else 1

    def get_img_base64(self):
        return self._img_base64

    def set_img_base64(self, value):
        self._img_base64 = value

    def get_hist(self):
        return self._hist

    def set_hist(self, value):
        self._hist = value

    def get_ticker(self):
        return self._ticker

    def set_ticker(self, value):
        self._ticker = value

    def get_scaler(self):
        return self._scaler

    def set_scaler(self, value):
        self._scaler = value

    def get_Model(self):
        return self._Model

    def set_Model(self, value):
        self._Model = value