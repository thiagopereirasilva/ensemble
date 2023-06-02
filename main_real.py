from MyEnsemble import MyEnsemble
from MyRegressor import MyRegressor
from MyDatasets import *
from river import metrics, stream
from Util import *
from tqdm import tqdm
import copy
import Util



SIMULATION_DATA_POINTS = 123

METRIC_WINDOW_SIZE = 100
METRIC_PREFERED_METRIC = 'Rolling_MAE'

COMBINATION_STRATEGY = Util.COMBINATION_STRATEGY_BEST_MODELS_AVERAGED
COMBINATION_STRATEGY_THRESHOLD = 0.9


def main():
    print('Reading dataset from csv file')
    # Y_LABEL = 'power_consumption'
    # SIMULATION_NAME = 'HouseholdPower' + "/" + str(SIMULATION_DATA_POINTS)
    # dataset = Household_Power_Consumption(SIMULATION_DATA_POINTS)

    # Y_LABEL = 'MedHouseVal'
    # SIMULATION_NAME = 'CaliforniaHousing' + "/" + str(SIMULATION_DATA_POINTS)
    # dataset = CaliforniaHousing()

    Y_LABEL = 'target'
    SIMULATION_NAME ='RandomRegression' + "/" + str(SIMULATION_DATA_POINTS)
    dataset = MakeRegression(n_samples=SIMULATION_DATA_POINTS, n_features=8, n_target=1, noise=0.2, y_label=Y_LABEL)
    print(dataset)
    # quit()

    print('****************\n')
    ensemble_metrics_dict = {
        'MAE': metrics.MAE(),
        'MSE': metrics.MSE(),
        'R2': metrics.R2(),
        'RMSE': metrics.RMSE(),
        'SMAPE': metrics.SMAPE(),
        'Rolling_MAE': metrics.Rolling(metrics.MAE(), window_size=METRIC_WINDOW_SIZE),
        'Rolling_MSE': metrics.Rolling(metrics.MSE(), window_size=METRIC_WINDOW_SIZE),
        'Rolling_R2': metrics.Rolling(metrics.R2(), window_size=METRIC_WINDOW_SIZE),
        'Rolling_RMSE': metrics.Rolling(metrics.RMSE(), window_size=METRIC_WINDOW_SIZE),
        'Rolling_SMAPE': metrics.Rolling(metrics.SMAPE(), window_size=METRIC_WINDOW_SIZE)
    }

    metrics_dict = {
        'MAE': metrics.MAE(),
        'MSE': metrics.MSE(),
        'R2': metrics.R2(),
        'RMSE': metrics.RMSE(),
        'SMAPE': metrics.SMAPE(),
        'Rolling_MAE': metrics.Rolling(metrics.MAE(), window_size=METRIC_WINDOW_SIZE),
        'Rolling_MSE': metrics.Rolling(metrics.MSE(), window_size=METRIC_WINDOW_SIZE),
        'Rolling_R2': metrics.Rolling(metrics.R2(), window_size=METRIC_WINDOW_SIZE),
        'Rolling_RMSE': metrics.Rolling(metrics.RMSE(), window_size=METRIC_WINDOW_SIZE),
        'Rolling_SMAPE': metrics.Rolling(metrics.SMAPE(), window_size=METRIC_WINDOW_SIZE)
    }

    linear_regression = get_LinearRegression(dataset.__class__.__name__)
    logistic = get_LogisticRegression(dataset.__class__.__name__)
    knn_regression = get_KNN_Model(2, dataset.__class__.__name__)
    adpt_rand_forest_regression = get_Adaptive_random_forest(
        10, dataset.__class__.__name__)
    ewa_regression = get_EWA_regressor(dataset.__class__.__name__)
    bagg_regression = get_BAGG(5, dataset.__class__.__name__)
    par = get_PARRegression(dataset.__class__.__name__)
    mlp = get_MLPRegression(dataset.__class__.__name__)
    hatr = get_HATR(dataset.__class__.__name__)
    htr = get_HTR(dataset.__class__.__name__)
    SRPR = get_SRPRegressor(
        n_models=3, dataset_name=dataset.__class__.__name__)
    SGT = get_SGTRegressor(dataset.__class__.__name__)

    # Populates an array with the Regression Models
    base_models = []

    base_models.append(MyRegressor(linear_regression,
                                   'Linear Regression', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(logistic,
                                   'Logistic Regression', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(adpt_rand_forest_regression,
                                   'Adaptive Random Forest', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(knn_regression,
                                   'KNN Regression', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(ewa_regression,
                                   'EWA', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(bagg_regression,
                                   'Bagging', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(par,
                                   'PAR Regression', copy.deepcopy(metrics_dict)))
    # base_models.append(MyRegressor(mlp,
    #                                'MLP Regression', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(hatr,
                                   'HAT Regression', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(htr,
                                   'HT Regression', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(SRPR,
                                   'SRP', copy.deepcopy(metrics_dict)))
    base_models.append(MyRegressor(SGT,
                                   'SGT', copy.deepcopy(metrics_dict)))

    start = time.time_ns()

    print("Creating Ensemble")
    ensemble = MyEnsemble(base_models=base_models,
                          metrics=ensemble_metrics_dict,
                          preferred_metric=METRIC_PREFERED_METRIC,
                          combination_strategy=COMBINATION_STRATEGY,
                          combination_threshold=COMBINATION_STRATEGY_THRESHOLD)
    print(ensemble)
    print('****************\n')


    print("Using dataset from csv file")
    print('****************\n')


    print('Configuring Simulation')
    print('\tData Points: {}\n\tPrefered Metric: {}\n\tWindow Size: {}\n\tCombination Strategy: {}\n\tCombination Threshold: {}'.format(
        SIMULATION_DATA_POINTS, METRIC_PREFERED_METRIC, METRIC_WINDOW_SIZE, COMBINATION_STRATEGY, COMBINATION_STRATEGY_THRESHOLD))
    print('****************\n')


    print('Running Simulation')
    run_simulation_dataset_pandas(
        ensemble, dataset, SIMULATION_DATA_POINTS, Y_LABEL)
    print('****************\n')


    print('Saving logs in csv format into folder \'logs/{}/\''.format(SIMULATION_NAME))
    # Save in csv file all base model's metrics
    for model in ensemble.get_base_models():
        model.save_log_csv(SIMULATION_NAME)

    # Save Ensemble's log into file
    ensemble.ensemble_save_log_csv(SIMULATION_NAME)

    # Saving the simulation params
    path = 'logs/'+SIMULATION_NAME+'/simulation.txt'
    with open(path, 'w') as f:
        f.write('\tData Points: {}\n\tPrefered Metric: {}\n\tWindow Size: {}\n\tCombination Strategy: {}\n\tCombination Threshold: {}\n\tEnsemble: {}'.format(
            SIMULATION_DATA_POINTS, METRIC_PREFERED_METRIC, METRIC_WINDOW_SIZE, COMBINATION_STRATEGY, COMBINATION_STRATEGY_THRESHOLD, ensemble))
    print('****************\n')


    print("Simulation finished. It took {:.2f} seconds to complete.".format(
        millis(start)/1000))


def run_simulation_dataset_pandas(ensemble, dataset_pandas, qtd_data_points, y_label):
    round = 0
    X = dataset_pandas
    y_label = X.pop(str(y_label))
    y_pred = 0
    for x, y in tqdm(stream.iter_pandas(X, y_label)):
        # for x, y in stream.iter_pandas(X, y_label):
        y_pred = ensemble.ensemble_predict_one(x)
        ensemble.ensemble_learn_one(x, y)
        # print(type(y_pred))
        # print(y_pred)
        ensemble.ensemble_update_metric(y, y_pred)

        ensemble.ensemble_log(round)
        round += 1
        if round > qtd_data_points:
            break


if __name__ == "__main__":
    main()
