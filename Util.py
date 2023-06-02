import time
from river import compose, ensemble, linear_model, metrics, neighbors, reco, facto,drift
from river import neural_net as nn
from river import optim, preprocessing, tree, datasets
from EvOAutoML import regression, pipelinehelper
from EvOAutoML.config import AUTOML_REGRESSION_PIPELINE, REGRESSION_PARAM_GRID
import itertools
from river import feature_extraction, stats, utils
import calendar
import numbers
import string
import math
import six
import numpy as np




COMBINATION_STRATEGY_MAJORITY_VOTING = 'Majority Voting Strategy'
COMBINATION_STRATEGY_WEIGHTED_AVERAGE = 'Weighted Average Strategy'
COMBINATION_STRATEGY_BEST_MODEL = 'Best Model Strategy'
COMBINATION_STRATEGY_BEST_MODELS_AVERAGED = 'Best Models Weighted Average Strategy'

MODEL_SEED = 42

def get_hour(x):
    x['hour'] = x['moment'].hour
    return x


def get_month(x):
    return {
        calendar.month_name[month]: month == x['month'].month
        for month in range(1, 13)
    }


def get_month_distances(x):
    return {
        calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2)
        for month in range(1, 13)
    }


def get_ordinal_date(x):
    return {'ordinal_date': x['month'].toordinal()}


def get_date_features(x):
    weekday = x['date'].weekday()
    return {'weekday': weekday, 'is_weekend': weekday in (5, 6)}


def millis(start_time):
    """
    Calculates the time spent in an interval
    Parameters in milliseconds.
    ----------
    name : start_time
      The initial time tick
    """
    elapsed = time.time_ns()-start_time
    return elapsed/1000000

# {'vendor_id': '2', 'pickup_datetime': datetime.datetime(2016, 1, 1, 0, 0, 17),
# 'passenger_count': 5, 'pickup_longitude': -73.98174285888672, 'pickup_latitude': 40.71915817260742,
# 'dropoff_longitude': -73.93882751464845, 'dropoff_latitude': 40.82918167114258, 'store_and_fwd_flag': 'N'}
def parse_Taxis(tr):
    tr["distance"] = (
        HaversineDistance(
            tr["pickup_latitude"], tr["pickup_longitude"], 
            tr["dropoff_latitude"], tr["dropoff_longitude"]
        )
    )
    tr["log_distance"] = np.log(tr['distance'])
    # tr["log_duration"] = np.log(tr['trip_duration'])
    tr['pickup_day_of_week'] = tr["pickup_datetime"].dt.week
    # day_name().astype('category')
    tr['pickup_month_of_year'] = tr["pickup_datetime"].dt.month_name().astype('category')
    tr['pickup_hour_of_day'] = tr["pickup_datetime"].dt.hour.astype('category')

    # Consider adding orders for categorical data to get the plots in nice way
    # tr['pickup_day_of_week'] = tr['pickup_day_of_week'].cat.set_categories(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], ordered=True)
    # tr['pickup_month_of_year'] = tr['pickup_month_of_year'].cat.set_categories(['January','February','March','April','May','June','July','August','September','October','November','December'], ordered=True)
    # tr['pickup_hour_of_day'] = tr['pickup_hour_of_day'].cat.set_categories(np.arange(0,24), ordered=True)

    return tr

def HaversineDistance(lat1,lon1,lat2,lon2):
	"""
	Returns the distance between two lat-long cordinates in km
	"""
	REarth = 6371
	lat = np.abs(np.array(lat1)-np.array(lat2))*np.pi/180
	lon = np.abs(np.array(lon1)-np.array(lon2))*np.pi/180
	lat1 = np.array(lat1)*np.pi/180
	lat2 = np.array(lat2)*np.pi/180
	a = np.sin(lat/2)*np.sin(lat/2)+np.cos(lat1)*np.cos(lat2)*np.sin(lon/2)*np.sin(lon/2)
	d = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
	d = REarth*d
	return d

def parse_MovieLens100k(x):
    # return {'user':int(x['user']), 'item':int(x['item'])}
    return {'user': 2.0}


def parse_Mv(x):
    # brown
    x3 = 0
    if x['x3'] == 'green':
        x3 = 1
    elif x['x3'] == 'red':
        x3 = 2

    x7 = 0
    if x['x7'] == 'yes':
        x7 = 1

    x8 = 0
    if x['x8'] == 'large':
        x8 = 1

    return {'x1': x['x1'], 'x2': x['x2'], 'x3': x3, 'x4': x['x4'], 'x5': x['x5'],
            'x6': x['x6'], 'x7': x7, 'x8': x8, 'x9': x['x9'], 'x10': x['x10']}


def get_preprocessing_pipeline(dataset_name):
    model_pipeline = compose.Pipeline()
    if dataset_name == datasets.Bikes().__class__.__name__:
        extract_features = compose.Select('clouds', 'humidity',
                                          'pressure', 'temperature', 'wind')
        extract_features += (
            get_hour |
            feature_extraction.TargetAgg(
                by=['station', 'hour'], how=stats.Mean())
        )

        model_pipeline |= extract_features
    elif dataset_name == datasets.AirlinePassengers().__class__.__name__:
        extract_features = compose.TransformerUnion(
            get_ordinal_date, get_month_distances)
        model_pipeline |= extract_features

    elif dataset_name == datasets.Restaurants().__class__.__name__:
        to_discard = ['store_id', 'date', 'genre_name',
                      'area_name', 'latitude', 'longitude']
        model = get_date_features
        for n in [7, 14, 21]:
            model += feature_extraction.TargetAgg(
                by='store_id', how=stats.RollingMean(n))
        model |= compose.Discard(*to_discard)
        model_pipeline |= model
        
    elif dataset_name == datasets.synth.Mv().__class__.__name__:
        model_pipeline = compose.FuncTransformer(parse_Mv)
    
    elif dataset_name == datasets.Taxis().__class__.__name__:
        model_pipeline = compose.FuncTransformer(parse_Taxis)
    
    elif dataset_name == datasets.MovieLens100K().__class__.__name__:
        to_discard = ['title', 'occupation', 'genres', 'gender', 'zip_code']
        model_pipeline = compose.Discard(*to_discard)

        model_pipeline = compose.Select('user', 'item')
        model_pipeline += (
            compose.Select('genres') | compose.FuncTransformer(split_genres)
        )
        model_pipeline+= (
            compose.Select('age') | compose.FuncTransformer(bin_age)
        )        
        # model_pipeline = preprocessing.PredClipper(
        #     regressor=model_pipeline,
        #     y_min=0,
        #     y_max=5
        # )
        print(model_pipeline)
    return model_pipeline

def split_genres(x):
    genres = x['genres'].split(', ')
    return {f'genre_{genre}': 1 / len(genres) for genre in genres}

def bin_age(x):
    if x['age'] <= 18:
        return {'age_0-18': 1}
    elif x['age'] <= 32:
        return {'age_19-32': 1}
    elif x['age'] < 55:
        return {'age_33-54': 1}
    else:
        return {'age_55-100': 1}


def get_LinearRegression(dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()

    learn = linear_model.LinearRegression(
        intercept_lr=0,
        optimizer=optim.SGD(0.001)
    )
    model |= scale | learn
    return model


def get_KNN_Model(n_neighbors, dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()
    learn = neighbors.KNNRegressor(
        window_size=672, n_neighbors=n_neighbors)
    model |= scale | learn
    return model


def get_Adaptive_random_forest(n_models, dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()
    learn = ensemble.AdaptiveRandomForestRegressor(n_models=n_models,
                                                   seed=MODEL_SEED,
                                                   metric=metrics.RMSE(),
                                                   binary_split=True,
                                                   max_depth=5
                                                   )
    model |= scale | learn
    return model


def get_EWA_regressor(dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()

    optimizers = [
        optim.SGD(0.01),
        optim.RMSProp(),
        optim.AdaGrad()
    ]

    learn = ensemble.EWARegressor([
        linear_model.LinearRegression(
            optimizer=o, intercept_lr=.1)
        for o in optimizers
    ],
        learning_rate=0.005
    )

    model |= scale | learn
    return model


def get_BAGG(n_models, dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()

    learn = ensemble.BaggingRegressor(
        model=linear_model.LinearRegression(intercept_lr=0.1),
        n_models=n_models,
        seed=MODEL_SEED
    )
    model |= scale | learn
    return model

def get_SRPRegressor(n_models, dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()
    base_model = tree.HoeffdingTreeRegressor(grace_period=50)
    learn = ensemble.SRPRegressor(
        model=base_model,
        training_method="patches",
        n_models=n_models,
        seed=MODEL_SEED,
        drift_detector=drift.ADWIN()
    )
    model |= scale | learn
    return model


def get_SGTRegressor(dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()

    learn = tree.SGTRegressor(
        delta=0.01,
        lambda_value=0.01,
        grace_period=20,
        feature_quantizer=tree.splitter.DynamicQuantizer(std_prop=0.1)
    )
    model |= scale | learn
    return model


def get_LogisticRegression(dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()

    learn = linear_model.LogisticRegression(
        optimizer=optim.SGD(.1))
    model |= scale | learn
    return model

def get_PARRegression(dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()

    learn = linear_model.PARegressor(
        C=0.01,
        mode=2,
        eps=0.1,
        learn_intercept=False
    )
    model |= scale | learn
    return model


# https://riverml.xyz/0.7.0/api/neural-net/MLPRegressor/
def get_MLPRegression(dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()

    learn = nn.MLPRegressor(
        # two hidden layers with 10 and 10 neurons, respectively.
        hidden_dims=(10, 10),
        activations=(
            nn.activations.Identity,
            nn.activations.Identity,
            nn.activations.Identity,
        ),
        optimizer=optim.SGD(1e-4),
        seed=MODEL_SEED
    )
    model |= scale | learn
    return model


def get_HATR(dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()

    learn = tree.HoeffdingAdaptiveTreeRegressor(
        grace_period=100,
        leaf_prediction='adaptive',
        model_selector_decay=0.5,
        max_depth=5,
        splitter=tree.splitter.TEBSTSplitter(),
        seed=MODEL_SEED
    )
    model |= scale | learn
    return model


def get_HTR(dataset_name):
    model = get_preprocessing_pipeline(dataset_name)
    scale = preprocessing.StandardScaler()

    learn = tree.HoeffdingTreeRegressor(
        grace_period=100,
        leaf_prediction='adaptive',
        model_selector_decay=0.5,
        max_depth=5,
        splitter=tree.splitter.TEBSTSplitter()
    )
    model |= scale | learn
    return model


def get_EVOAutoML():
    model_pipeline = compose.Pipeline(
        ('Scaler', pipelinehelper.PipelineHelperTransformer([
            ('StandardScaler', preprocessing.StandardScaler()),
            # ('MinMaxScaler', preprocessing.MinMaxScaler()),
            # ('MinAbsScaler', preprocessing.MaxAbsScaler()),
            # ('OneHotEnconder', preprocessing.OneHotEncoder())
        ])),
        ('Regressor', pipelinehelper.PipelineHelperRegressor([
            ('HT', tree.HoeffdingTreeRegressor()),
            ('KNN', neighbors.KNNRegressor()),
            ('HATR', tree.HoeffdingAdaptiveTreeRegressor()),
            ('PAR', linear_model.PARegressor()),
            ('LR', linear_model.LinearRegression())
        ]))
    )
    EvOAutoML_model = regression.EvolutionaryBaggingRegressor(
        model=model_pipeline,
        param_grid={
            'Scaler': model_pipeline.steps['Scaler'].generate({}),
            # 'Regressor': AUTOML_REGRESSION_PIPELINE.steps['Regressor'].generate({
            'Regressor': model_pipeline.steps['Regressor'].generate({
                'HT__binary_split': [True, False],
                'HT__max_depth': [10, 30, 60, 100],
                'HT__grace_period': [10, 100, 200],
                'HT__max_size': [5, 10],
                'HT__leaf_prediction': ['adaptive', 'mean'],

                'KNN__n_neighbors': [1, 5, 20],
                'KNN__window_size': [100, 500, 1000],
                'KNN__p': [1, 2, 5],

                'HATR__max_size': [5, 10],
                'HATR__grace_period': [10, 100, 200],
                'HATR__leaf_prediction': ['adaptive', 'mean'],
                'HATR__model_selector_decay': [0.1, 0.5],

                'PAR__mode': [1, 2],

                'LR__l2': [.0, .01, .001],
            })
        },
        seed=42
    )
    return EvOAutoML_model
