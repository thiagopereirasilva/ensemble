from river import metrics
import pandas as pd
from datetime import datetime
import time
import os


class MyRegressor:
    """
    A class that represents an Regression Model

    Attributes
    ----------
    model : Model
        a regression model
    name : str
        the name of the model
    metrics : dictionary
        a dictionary of metrics used to evaluate the model
    log : Pandas Dataframe
        the log of events
    """
    model = None
    name = ""
    metrics = {}
    last_predicted_value = 0.0
    predictions = []
    # log_columns = {
    #     'Round': [],
    #     'Time2predict': [],
    #     'Time2learn': [],
    #     'MAE': [],
    #     'MSE': [],
    #     'R2': [],
    #     'Rolling_MAE': [],
    #     'Rolling_MSE': [],
    #     'Rolling_R2': [],
    #     'Algorithm': [],
    # }
    log_columns = {
        'Round': [],
        'Timestamp': [],
        'Time2predict': [],
        'Time2learn': [],
        'MAE': [],
        'MSE': [],
        'R2': [],
        'RMSE': [],
        'SMAPE': [],
        # 'RMSLE': [],
        'Rolling_MAE': [],
        'Rolling_MSE': [],
        'Rolling_R2': [],
        'Rolling_RMSE': [],
        'Rolling_SMAPE': [],
        # 'Rolling_RMSLE': [],
        'Algorithm': []
    }

    log = pd.DataFrame(columns=log_columns)
    log_buffer = []
    # window_size = 10

    def __init__(self, model, name, metrics):
        self.model = model
        self.name = name
        self.metrics = metrics
        self.last_predicted_value = 0.0
        self.last_time2predict = 0.0
        self.last_time2train = 0.0
        self.predictions = []
        # self.log = pd.DataFrame(self.log_columns, columns=['Round', 'timestamp', 'Time2predict', 'Time2learn',
        #    'MAE', 'MSE', 'R2', 'Rolling_MAE', 'Rolling_MSE',
        #    'Rolling_R2', 'Algorithm'])
        self.log.set_index('Round')
        self.log_buffer = []

    def set_last_time2predict(self, time):
        self.last_time2predict = time

    def get_last_time2predict(self):
        return self.last_time2predict

    def set_last_time2train(self, time):
        self.last_time2train = time

    def get_last_time2train(self):
        return self.last_time2train

    def get_last_predicted_value(self):
        return self.last_predicted_value

    def set_last_predicted_value(self, value):
        self.last_predicted_value = value

    def add_predicted_value(self, value):
        self.predictions.insert(len(self.predictions), value)

    def get_predictions(self):
        return self.predictions.copy()

    def get_log(self):
        self.log = self.flush_events(self.log, self.log_buffer)
        return self.log

    # def log_event(self, round, timestamp, time_predict, time_learn, metrics_dict, algorithm):
    #     dic = {'Round': round, 'timestamp': timestamp, 'Time2predict': time_predict,
    #            'Time2learn': time_learn, 'Algorithm': algorithm}
    #     dic.update(metrics_dict)
    #     self.log = self.log.append(dic.copy(), ignore_index=True)

    def log_event(self, round, timestamp, time_predict, time_learn, metrics_dict, algorithm):
        row = [
            round,
            timestamp,
            time_predict,
            time_learn
        ]
        for i in list(metrics_dict.values()):
            row.append(i)
        row.append(algorithm)

        self.log_buffer.insert(0, row)

    def flush_events(self, df, df_buffer):
        if len(df_buffer) > 0:
            df = pd.concat(
                [pd.DataFrame(df_buffer, columns=self.log_columns), df], ignore_index=True)
            df_buffer.clear()
        return df

    def save_log_csv(self, simulation_name):
        os.makedirs('logs/{}/'.format(simulation_name), exist_ok=True)
        self.log = self.get_log()
        self.log.to_csv('logs/{}/'.format(simulation_name) +
                        self.name+'.csv', index=False)

    def get_model(self):
        return self.model

    def get_metrics(self):
        return self.metrics

    def get_metric_value(self, metric_name):
        return float(self.metrics[metric_name].get())

    def set_model(self, model):
        self.model = model

    def set_metric(self, name, metric):
        self.metrics[name] = metric

    def get_name(self):
        return self.name
