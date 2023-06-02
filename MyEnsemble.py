from river import (compose, datasets, linear_model, metrics, preprocessing,
                   stream)

from Util import millis
import Util
import MyRegressor
import time
import pandas as pd
import os
from numba import jit, cuda, njit
import cython
import copy

class MyEnsemble:
    """
    A class that represents an Ensemble

    Attributes
    ----------
    models : MyRegressor
        a list of regression models
    metrics : dictionary
        a dictionary of metrics used to evaluate the model
    """
    ensemble_base_models = []
    # ensemble_metrics = {}
    ensemble_preferred_metric = ''
    ensemble_log_columns = {
        'Round': [],
        'Timestamp': [],
        'Time2predict': [],
        'Time2learn': [],
        'MAE': [],
        'MSE': [],
        'R2': [],
        'RMSE': [],
        'SMAPE': [],
        'Rolling_MAE': [],
        'Rolling_MSE': [],
        'Rolling_R2': [],
        'Rolling_RMSE': [],
        'Rolling_SMAPE': [],
        'Algorithm': []
    }
    ensemble_log_buffer = []
    ensemble_elog = pd.DataFrame(columns=ensemble_log_columns)

    ensemble_last_predicted_value = 0
    ensemble_last_time2predict = 0
    ensemble_last_time2train = 0
    ensemble_last_algorithm = ""

    ensemble_combination_strategy = ''
    ensemble_combination_threshold = 0

    def __init__(self, base_models, metrics, preferred_metric='MAE',
                 combination_strategy=Util.COMBINATION_STRATEGY_BEST_MODEL,
                 combination_threshold=0):
        self.ensemble_base_models = base_models
        self.ensemble_metrics = metrics
        # print(metrics)
        self.ensemble_preferred_metric = preferred_metric
        self.ensemble_combination_strategy = combination_strategy
        self.ensemble_combination_threshold = combination_threshold
        self.ensemble_last_algorithm = ""
        self.ensemble_elog.set_index('Round')
        self.ensemble_log_buffer = []

        self.ensemble_last_predicted_value = 0
        self.ensemble_last_time2predict = 0
        self.ensemble_last_time2train = 0

    def get_base_models(self):
        return self.ensemble_base_models

    # @jit(nopython=False, forceobj=True, nogil=True, cache=True, parallel=True, forceinline=True, fastmath=True)
    # @cuda.jit(device=True, inline=False, fastmath=True, opt=True)
    @cython.cfunc
    def ensemble_learn_one(self, x, y):
        """
        Train all base learners that compose the ensemble.
        ----------
        x : features
            The features
        y : target
            The ground truth
        """
        ensemble_time_inst = time.time_ns()
        for regressor in self.ensemble_base_models:
            time_inst = time.time_ns()
            regressor.get_model().learn_one(x, y)
            time2learn = millis(time_inst)
            regressor.set_last_time2train(time2learn)
        ensemble_time2learn = millis(ensemble_time_inst)
        self.ensemble_last_time2train = ensemble_time2learn

    @cython.cfunc
    def ensemble_predict_one(self, x):
        ensemble_time_inst = time.time_ns()

        # Make predictions in all base models
        for regressor in self.ensemble_base_models:
            time_inst = time.time_ns()
            y_pred = regressor.get_model().predict_one(x)
            # y_pred = MyEnsemble.predict_paral.remote(regressor, x)
            # y_pred = ray.get(y_pred)
            time2predict = millis(time_inst)
            regressor.set_last_predicted_value(y_pred)
            regressor.set_last_time2predict(time2predict)

        self.ensemble_last_time2predict = millis(ensemble_time_inst)

        regressor = 0
        if self.ensemble_combination_strategy == Util.COMBINATION_STRATEGY_BEST_MODEL:
            # Choose the current best model according to preferred metric
            regressor = self.get_best_current_base_model(
                self.ensemble_preferred_metric)
            self.ensemble_last_predicted_value = regressor.get_last_predicted_value()
            return regressor.get_last_predicted_value()
        elif self.ensemble_combination_strategy == Util.COMBINATION_STRATEGY_MAJORITY_VOTING:
            average_pred = 0
            for model in self.ensemble_base_models:
                average_pred += model.get_last_predicted_value()
            aux = average_pred / len(self.ensemble_base_models)
            # return aux if aux > 0 else 0
            return aux
        elif self.ensemble_combination_strategy == Util.COMBINATION_STRATEGY_WEIGHTED_AVERAGE:
            weight_average_pred = 0
            sum_metric_value = 0
            for model in self.ensemble_base_models:
                m = model.get_metrics()[self.ensemble_preferred_metric].get()
                sum_metric_value += m
                weight_average_pred += (model.get_last_predicted_value() * m)
            aux = weight_average_pred / sum_metric_value
            return aux
            # return aux if aux > 0 else 0
        elif self.ensemble_combination_strategy == Util.COMBINATION_STRATEGY_BEST_MODELS_AVERAGED:
            weight_average_pred = 0
            sum_metric_value = 0
            threshold_satisfied  = False
            for model in self.ensemble_base_models:
                m = model.get_metrics()[self.ensemble_preferred_metric].get()
                if model.get_metrics()[self.ensemble_preferred_metric].bigger_is_better == True:
                    if m > self.ensemble_combination_threshold:
                        threshold_satisfied  = True
                        sum_metric_value += m
                        weight_average_pred += (
                            model.get_last_predicted_value() * m)
                else:
                    if m < self.ensemble_combination_threshold:
                        threshold_satisfied  = True
                        sum_metric_value += m
                        weight_average_pred += (
                            model.get_last_predicted_value() * m)
            # If there are no model with metric below a given threshold, than use the best model strategy.
            if threshold_satisfied == False:
                regressor = self.get_best_current_base_model(
                    self.ensemble_preferred_metric)
                # print(Util.COMBINATION_STRATEGY_BEST_MODEL)
                self.ensemble_last_predicted_value = regressor.get_last_predicted_value()
                return regressor.get_last_predicted_value()
            aux = weight_average_pred / sum_metric_value
            # print(Util.COMBINATION_STRATEGY_BEST_MODELS_AVERAGED)
            return aux
            # return aux if aux > 0 else 0

    @ cython.cfunc
    def get_best_current_base_model(self, metric_name):
        """
        This method returns the best model according to a specified metric
        """
        best_model = self.ensemble_base_models[0]
        best_model_metric_value = float('inf')

        # Indicate if a high value is better than a low one or not.
        if self.ensemble_base_models[0].get_metrics()[metric_name].bigger_is_better == True:
            best_model_metric_value = float('-inf')

        # Iterates over all base models
        for model in self.ensemble_base_models:
            metric = model.get_metrics()[metric_name]

            # Indicate if a high value is better than a low one or not.
            if metric.bigger_is_better == True:
                if model.get_metric_value(metric_name) > best_model_metric_value:
                    # print('big is better')
                    best_model = model
                    best_model_metric_value = float(metric.get())
            elif metric.bigger_is_better == False:
                if model.get_metric_value(metric_name) < best_model_metric_value:
                    # print('big is not better')
                    best_model = model
                    best_model_metric_value = float(metric.get())
        self.ensemble_last_algorithm = best_model.get_name()
        return best_model

    def get_average_predicted_value(self, models):
        """
        This method returns the average of the basel model's predictions
        See more in:
        https://machinelearningmastery.com/weighted-average-ensemble-with-python/#:~:text=Weighted%20average%20or%20weighted%20sum%20ensemble%20is%20an%20ensemble%20machine,related%20to%20the%20voting%20ensemble.
        """
        average_pred = 0

        for model in models:
            average_pred = average_pred + model.get_predicted_value()
            aux = average_pred / len(models)
        return aux if aux > 0 else 0


    def ensemble_update_metric(self, y, y_pred):
        # First updates ensemble metrics
        for name, metric in self.ensemble_metrics.items():
            # print('updating ', name, metric, y_pred, y)
            metric = metric.update(y, y_pred)
            # print(metric.get())

        # Then updates every base model's metrics
        self.base_models_update_metric(y, y_pred)

    @ cython.cfunc
    def base_models_update_metric(self, y, y_pred):
        for regressor in self.ensemble_base_models:
            metrics = regressor.get_metrics()
            y_pred_model = regressor.get_last_predicted_value()
            for name, metric in metrics.items():
                metric.update(y, y_pred_model)

    # def ensemble_get_metrics(self):
    #     return self.ensemble_metrics

    def __str__(self):

        str_models = ""
        for model in self.ensemble_base_models:
            str_models += str(model.get_name()) + ','
        aux = 'MyEmsemble\n \tBase Learners (qtd): {}\n \tPerformance Metric: {}\n \tBase Learners: {}'.format(
            len(self.ensemble_base_models), self.ensemble_preferred_metric, str_models)
        return aux

    @ cython.cfunc
    def ensemble_log_event(self, round, timestamp, time_predict, time_learn, metrics_dict, algorithm):
        # print('log event')
        row = [
            round,
            timestamp,
            time_predict,
            time_learn
        ]
        for i in list(metrics_dict.values()):
            value = i
            # value = i
            # print(value)
            row.append(value)
        row.append(algorithm)

        self.ensemble_log_buffer.insert(0, row)

    def ensemble_log(self, round):
        # Log ensemble
        metrics_dict = {}
        # print(self.ensemble_metrics.items())
        for name, metric in self.ensemble_metrics.items():
            metrics_dict.update({name: float(metric.get())})
            # print(metric)
        self.ensemble_log_event(round=round,
                                timestamp=time.time_ns(),
                                time_predict=self.ensemble_last_time2predict,
                                time_learn=self.ensemble_last_time2train,
                                metrics_dict=metrics_dict,
                                algorithm=self.ensemble_last_algorithm)

        # Log all base model
        for regressor in self.ensemble_base_models:
            metrics = regressor.get_metrics()
            metrics_dict = {}
            for name, metric in metrics.items():
                metrics_dict.update({name: float(metric.get())})
            regressor.log_event(round=round,
                                timestamp=time.time_ns(),
                                time_predict=regressor.get_last_time2predict(),
                                time_learn=regressor.get_last_time2train(),
                                metrics_dict=metrics_dict,
                                algorithm=regressor.name)

    def ensemble_get_log(self):
        self.ensemble_elog = self.flush_events(
            self.ensemble_elog, self.ensemble_log_buffer)
        return self.ensemble_elog

    def ensemble_save_log_csv(self, simulation_name):
        self.ensemble_elog = self.ensemble_get_log()
        os.makedirs('logs/{}/'.format(simulation_name), exist_ok=True)
        self.ensemble_elog.to_csv(
            'logs/{}/Ensemble.csv'.format(simulation_name), index=False)

    def flush_events(self, df, df_buffer):
        if len(df_buffer) > 0:
            df = pd.concat(
                [pd.DataFrame(df_buffer, columns=self.ensemble_log_columns), df], ignore_index=True)
            df_buffer.clear()
        return df
