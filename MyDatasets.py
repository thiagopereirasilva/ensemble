import pandas as pd
from river import stream
from collections import Counter
import glob
import statistics
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_california_housing, make_regression

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression
def MakeRegression(n_samples, n_features, n_target, noise, y_label):
    X, Y = make_regression(n_samples=n_samples, 
                                 n_features=n_features, 
                                 n_informative=int(n_features/2), 
                                 n_targets=n_target, 
                                 bias=0.3, 
                                 effective_rank=2, 
                                 tail_strength=0.5, 
                                 noise=noise, 
                                 shuffle=True, 
                                 coef=False, 
                                 random_state=None)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y, columns=[y_label])
    dataset = pd.concat([X, Y], axis=1, join='inner')
    return dataset



# https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
# 20640 observations
def CaliforniaHousing():
    dataset = fetch_california_housing(return_X_y=False, as_frame=True)
    data =  pd.concat([dataset.data, dataset.target], axis=1, join='inner')
    return data


# Household Electric Power Consumption (http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
# 3.global_active_power: household global minute-averaged active power (in kilowatt)
# 4.global_reactive_power: household global minute-averaged reactive power (in kilowatt)
# 5.voltage: minute-averaged voltage (in volt)
# 6.global_intensity: household global minute-averaged current intensity (in ampere)
# 7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
# 8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
# 9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.
# y = power_consumption
def Household_Power_Consumption(size):
    df = pd.read_csv('./datasets/household_power_consumption.txt', sep=';',
                     parse_dates={'dt': ['Date', 'Time']}, infer_datetime_format=True,
                     low_memory=False, na_values=['nan', '?'], index_col='dt')

    # filling nan with mean in any columns
    for j in range(0, 7):
        df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())

    eq1 = (df['Global_active_power']*1000/60)
    eq2 = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
    df['power_consumption'] = eq1 - eq2

    df['Date'] = df.index.date
    df['time'] = df.index.time

    df['Date'] = df['Date'].astype(str)
    df['time'] = df['time'].astype(str)

    df['exact_time'] = df['Date']+";"+df['time']
    df['exact_time_DT'] = pd.to_datetime(df['exact_time'],format="%Y-%m-%d;%H:%M:%S")
    # df = df.groupby(pd.Grouper(key='exact_time_DT',freq='d')).sum()

    df['hour'] = df.index.strftime('%H')
    df['day'] = df.index.strftime('%d')
    df['month'] = df.index.strftime('%m')

    df['hour'] = df['hour'].astype(int)
    df['day'] = df['day'].astype(int)
    df['month'] = df['month'].astype(int)

    data = df.drop(['Date', 'time','exact_time', 'exact_time_DT'],axis = 1)
    data = data.iloc[:size]
    # print(data.dtypes)
    # print(data.shape)
    return data

# y = PM2.5
def Beijing_Air_Quality():
    filePath = './datasets/beijin'
    allFiles = glob.glob(filePath + "/*.csv")
    dataFrames = []
    for i in allFiles:
        df = pd.read_csv(i, index_col=None, header=0)
        dataFrames.append(df)
    data = pd.concat(dataFrames)
    data.drop(["No"], axis=1, inplace=True)
    data.rename(columns = {'year': 'Year',
                       'month': 'Month',
                       'day': "Day",
                       'hour': 'Hour',
                       'pm2.5': 'PM2.5',
                       'DEWP': 'DewP',
                       'TEMP': 'Temp',
                       'PRES': 'Press',
                       'RAIN': 'Rain',
                       'wd': 'WinDir',
                       'WSPM': 'WinSpeed',
                       'station': 'Station'}, inplace = True)

    # fill the null values in numerical columns with average specific to certain column
    # fill in the missing data in the columns according to the Month average.
    unique_Month = pd.unique(data.Month)

    # find PM2_5 averages in Month specific
    # Equalize the average PM2.5 values to the missing values in PM2_5 specific to Month
    temp_data = data.copy()  # set temp_data variable to avoid losing real data
    columns = ["PM2.5", 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'Temp', 'Press', 'DewP', 'Rain', 'WinSpeed'] # it can be add more column
    for c in unique_Month:
        
        # create Month filter
        Month_filtre = temp_data.Month == c
        # filter data by Month
        fitered_data = temp_data[Month_filtre]
    
        # find average for PM2_5 in specific to Month
        for s in columns:
            mean = np.round(np.mean(fitered_data[s]), 2)
            if ~np.isnan(mean): # if there if average specific to Month
                fitered_data[s] = fitered_data[s].fillna(mean)
                # print(f"Missing Value in {s} column fill with {mean} when Month:{c}")
            else: # find average for all data if no average in specific to Month
                all_data_mean = np.round(np.mean(data[s]),2)
                fitered_data[s] = fitered_data[s].fillna(all_data_mean)
                # print(f"Missing Value in {s} column fill with {all_data_mean}")
        # Synchronize data filled with missing values in PM2.5 to data temporary            
        temp_data[Month_filtre] = fitered_data

    # equate the deprecated temporary data to the real data variable
    data = temp_data.copy()

    # fill the null values in categorical columns with mode specific to certain column

    # fill in the missing data in the WinDir column with mode values according to the Station.
    unique_Station = pd.unique(data.Station)

    # find columns mode value in WinDir column according to Station column specific
    # Equalize the mode values of columns to the missing values
    temp_data = data.copy()  # set temp_data variable to avoid losing real data
    columns = ["WinDir"] # it can be add more column
    for c in unique_Station:
        
        # create Station filter
        Station_filtre = temp_data.Station == c
        
        # filter data by Station
        filtered_data = temp_data[Station_filtre]
        
        # find mode for WinDir specific to Station
        for column in columns:
            mode = statistics.mode(filtered_data[column])
            filtered_data[column] = filtered_data[column].fillna(mode)
            # print(f"Missing Value in {column} column fill with {mode} when Station:{c}")

        # Synchronize data filled with missing values in WinDir to data temporary            
        temp_data[Station_filtre] = filtered_data

    # equate the deprecated temporary data to the real data variable
    data = temp_data.copy()
    categorical_variables = ["WinDir", "Station"]
    for i in categorical_variables:
        # print(f"For {i} column ")
        data[f"{i}"] = labelEncoder(data[f"{i}"])
        # print("**********************************")
    return data


# define a function for label encoding
def labelEncoder(labelColumn):
    labelValues = labelColumn
    unique_labels = labelColumn.unique()
    le = LabelEncoder()
    labelColumn = le.fit_transform(labelColumn)
    # print('Encoding Approach:')
    # for i, j in zip(unique_labels, labelColumn[np.sort(np.unique(labelColumn, return_index=True)[1])]): 
    #     print(f'{i}  ==>  {j}')
    return labelColumn


# data = Beijing_Air_Quality()
# print(data.shape)
# print(data.head())