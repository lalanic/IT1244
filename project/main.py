import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.max_rows', None)


is_train_model = True  # False = Use pretrained model
is_by_symbol = True  # True = Train by company, False = Train by sector
is_plot_line = True  # True = Print root-mean-square errors for all 4 y-values and plot predicted vs actual line graph


COMPANY_INDEX = 99  # Change company
SECTOR_INDEX = 4  # Change industry (sector)

days_backtracked = 1  # Change number of previous days data used
train_ratio = 0.8
val_ratio = 0.1
num_of_features = 7

initial_label = 0
end_label = 4
num_of_labels = end_label - initial_label

LSTM_units = 128
num_of_epochs = 25


def load_data(filename):
    """
    Loads parquet file into a pandas dataframe.

    :param filename: Raw string of dataset in parquet filepath
    :return: Pandas Dataframe of dataset
    """
    dataset = pd.read_parquet(filename)
    return dataset


def clean_dataframe(dataframe):
    """
    1) Removes companies with non-standard number of rows. (CEG, OGN)
    2) Removes unused columns containing strings. (Security and GICS Sub-Industry)
    3) Shifts column positions to leftmost. (Symbol and GICS Sector)

    :param dataframe: Pandas Dataframe of dataset
    :return: Modified Pandas Dataframe of dataset
    """
    cleaned_dataframe = dataframe.copy(deep=True)
    cleaned_dataframe = cleaned_dataframe[cleaned_dataframe.Symbol != "CEG"]
    cleaned_dataframe = cleaned_dataframe[cleaned_dataframe.Symbol != "OGN"]
    cleaned_dataframe = cleaned_dataframe.drop(["Security", "GICS Sub-Industry"], axis=1)
    cleaned_dataframe = cleaned_dataframe.reset_index()

    column_to_move = cleaned_dataframe.pop("Symbol")
    cleaned_dataframe.insert(0, "Symbol", column_to_move)

    column_to_move = cleaned_dataframe.pop("GICS Sector")
    cleaned_dataframe.insert(0, "Sector", column_to_move)

    return cleaned_dataframe


def process_data(dataframe):
    """
    Adds feature columns depending on days_backtracked and returns a list of pandas dataframes by company.

    :param dataframe: Pandas Dataframe of dataset
    :return: List of modified pandas dataframes grouped by company
    """

    grouped_by_symbol = dataframe.groupby(dataframe["Symbol"])

    dataframe_list_by_symbol = []
    for name, data in grouped_by_symbol:
        dataframe_list_by_symbol.append(data)

    for i in range(len(dataframe_list_by_symbol)):
        dataframe_list_by_symbol[i] = dataframe_list_by_symbol[i].reset_index(drop=True)
        for j in range(days_backtracked, 0, -1):
            dataframe_list_by_symbol[i]["D-" + str(j) + " Open"] = dataframe_list_by_symbol[i]["Open"].shift(periods=j, axis=0)
            dataframe_list_by_symbol[i]["D-" + str(j) + " High"] = dataframe_list_by_symbol[i]["High"].shift(periods=j, axis=0)
            dataframe_list_by_symbol[i]["D-" + str(j) + " Low"] = dataframe_list_by_symbol[i]["Low"].shift(periods=j, axis=0)
            dataframe_list_by_symbol[i]["D-" + str(j) + " Close"] = dataframe_list_by_symbol[i]["Close"].shift(periods=j, axis=0)
            # dataframe_list_by_symbol[i]["D-" + str(j) + " Adj Close"] = dataframe_list_by_symbol[i]["Adj Close"].shift(periods=j, axis=0)
            # dataframe_list_by_symbol[i]["D-" + str(j) + " Vol"] = dataframe_list_by_symbol[i]["Volume"].shift(periods=j, axis=0)
            # dataframe_list_by_symbol[i]["D-" + str(j) + " News Vol Proportion"] = (dataframe_list_by_symbol[i]["News - Volume"] / dataframe_list_by_symbol[i]["News - All News Volume"]).shift(periods=j, axis=0)
            # dataframe_list_by_symbol[i]["D-" + str(j) + " Net News Sentiment"] = (dataframe_list_by_symbol[i]["News - Positive Sentiment"] - dataframe_list_by_symbol[i]["News - Negative Sentiment"]).shift(periods=j, axis=0)
            dataframe_list_by_symbol[i]["D-" + str(j) + " Pos News Sentiment"] = dataframe_list_by_symbol[i]["News - Positive Sentiment"].shift(periods=j, axis=0)
            dataframe_list_by_symbol[i]["D-" + str(j) + " Layoffs"] = dataframe_list_by_symbol[i]["News - Layoffs"].shift(periods=j, axis=0)
            dataframe_list_by_symbol[i]["D-" + str(j) + " Adverse"] = dataframe_list_by_symbol[i]["News - Adverse Events"].shift(periods=j, axis=0)
        dataframe_list_by_symbol[i].dropna(inplace=True, ignore_index=True)

    return dataframe_list_by_symbol


def convert_parquet_to_dataframe_list(filename):
    dataframe = load_data(filename)
    cleaned_dataframe = clean_dataframe(dataframe)
    dataframe_list_by_symbol = process_data(cleaned_dataframe)

    return dataframe_list_by_symbol


def normalize_by_symbol(dataframe_list_by_symbol):
    """
    Normalizes x-values by company with scaling parameters obtained from training data.

    :param dataframe_list_by_symbol: List of pandas dataframes grouped by company
    :return: List of pandas dataframes grouped by company with x-values normalized by company
    """
    num_of_companies = len(dataframe_list_by_symbol)
    num_of_rows = len(dataframe_list_by_symbol[0])
    num_of_train_rows = int(train_ratio * (1 - val_ratio) * num_of_rows)

    df_copy_list = []
    for i in range(num_of_companies):
        df_copy_list.append(dataframe_list_by_symbol[i].copy(deep=True))

    for i in range(num_of_companies):
        for j in range(25, len(df_copy_list[i].columns)):
            col_min = df_copy_list[i].iloc[:num_of_train_rows, j].min()
            col_max = df_copy_list[i].iloc[:num_of_train_rows, j].max()
            diff = col_max - col_min
            if diff != 0:
                df_copy_list[i].iloc[:, j] -= col_min
                df_copy_list[i].iloc[:, j] /= diff

    return df_copy_list


# List of df grouped by symbol -> 1 full df + 1 train-only df -> Group both full and train-only df by sectors
# -> Scaling parameters from train-only df applied to full df -> Full df recombined and grouped by symbol
def normalize_by_sector(dataframe_list_by_symbol):
    """
    Normalizes x-values by sector with scaling parameters obtained from training data.

    :param dataframe_list_by_symbol: List of pandas dataframes grouped by company
    :return: List of pandas dataframes grouped by company with x-values normalized by sector
    """
    num_of_companies = len(dataframe_list_by_symbol)
    num_of_rows = len(dataframe_list_by_symbol[0])
    num_of_train_rows = int(train_ratio * (1 - val_ratio) * num_of_rows)

    df_copy_list = []
    for i in range(num_of_companies):
        df_copy_list.append(dataframe_list_by_symbol[i].copy(deep=True))

    combined_df = pd.DataFrame()
    for i in range(num_of_companies):
        combined_df = pd.concat([combined_df, df_copy_list[i]])

    combined_train_df = pd.DataFrame()
    for i in range(num_of_companies):
        combined_train_df = pd.concat([combined_train_df, df_copy_list[i].iloc[:num_of_train_rows, :]])

    df_list_by_sector = group_by_sector(combined_df)
    train_df_list_by_sector = group_by_sector(combined_train_df)
    num_of_sectors = len(train_df_list_by_sector)

    for i in range(num_of_sectors):
        for j in range(25, len(df_copy_list[i].columns)):
            col_min = train_df_list_by_sector[i].iloc[:, j].min()
            col_max = train_df_list_by_sector[i].iloc[:, j].max()
            diff = col_max - col_min
            if diff != 0:
                df_list_by_sector[i].iloc[:, j] -= col_min
                df_list_by_sector[i].iloc[:, j] /= diff

    recombined_df = pd.DataFrame()
    for i in range(num_of_sectors):
        recombined_df = pd.concat([recombined_df, df_list_by_sector[i]])

    df_list = group_by_symbol(recombined_df)
    return df_list


def split_train_test(scaled_dataframe_list):
    """
    Splits list of feature scaled pandas dataframes into train and test dataframes.

    :param scaled_dataframe_list: List of feature scaled pandas dataframes grouped by company
    :return: Pandas dataframe with training data, Pandas dataframe with testing data
    """
    num_of_companies = len(scaled_dataframe_list)
    num_of_rows = len(scaled_dataframe_list[0])
    num_of_train_rows = int(train_ratio * num_of_rows)

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i in range(num_of_companies):
        train_df = pd.concat([train_df, scaled_dataframe_list[i].iloc[:num_of_train_rows, :]])
        test_df = pd.concat([test_df, scaled_dataframe_list[i].iloc[num_of_train_rows:, :]])

    return train_df, test_df


def group_by_symbol(dataframe):
    """
    Returns a list of pandas dataframes grouped by company.

    :param dataframe: Pandas dataframe containing dataset
    :return: List of pandas dataframes grouped by company
    """
    grouped_by_symbol = dataframe.groupby(dataframe["Symbol"])
    dataframe_list_by_symbol = []
    for name, data in grouped_by_symbol:
        dataframe_list_by_symbol.append(data)

    return dataframe_list_by_symbol


def group_by_sector(dataframe):
    """
    Returns a list of pandas dataframes grouped by sector.

    :param dataframe: Pandas dataframe containing dataset
    :return: List of pandas dataframes grouped by sector
    """
    grouped_by_sector = dataframe.groupby(dataframe["Sector"])
    dataframe_list_by_sector = []
    for name, data in grouped_by_sector:
        dataframe_list_by_sector.append(data)

    return dataframe_list_by_sector


def convert_to_lstm_input(dataset_single):
    """
    Converts pandas dataframe into numpy array to input into LSTM model.

    :param dataset_single: Pandas dataframe containing data of a single company or sector
    :return: 3-D numpy array of x-values, 2-D numpy array of y-values
    """
    dataset_single = dataset_single.iloc[:, 3:]
    dataset_single_as_np = dataset_single.to_numpy()
    x = []
    y = []

    num_of_rows = len(dataset_single_as_np)
    for i in range(num_of_rows):
        features_by_day = []
        for j in range(days_backtracked):
            offset = j * num_of_features
            features_by_day.append(dataset_single_as_np[i, (22 + offset):(22 + num_of_features + offset)])
        x.append(features_by_day)
        y.append(dataset_single_as_np[i, initial_label:end_label])

    x_as_np = np.array(x)
    y_as_np = np.array(y)

    return x_as_np, y_as_np


def create_model(shape):
    """
    Creates a neural network model.

    :param shape: Shape of input array
    :return: tf.keras.Model object
    """
    lstm_model = tf.keras.models.Sequential()

    lstm_model.add(tf.keras.layers.InputLayer(input_shape=shape))

    lstm_model.add(tf.keras.layers.LSTM(units=LSTM_units, return_sequences=True))
    lstm_model.add(tf.keras.layers.LSTM(units=LSTM_units))
    # lstm_model.add(tf.keras.layers.Dense(units=LSTM_units, kernel_initializer="lecun_normal", activation="selu"))
    lstm_model.add(tf.keras.layers.Dense(units=LSTM_units, activation="relu"))
    lstm_model.add(tf.keras.layers.Dense(units=LSTM_units))
    lstm_model.add(tf.keras.layers.Dense(units=LSTM_units))

    lstm_model.add(tf.keras.layers.Dense(units=num_of_labels, activation="linear"))

    return lstm_model


def fit_model(model, x_train, y_train):
    """
    Trains the neural network model on a single company.

    :param model: tf.keras.Model object
    :param x_train: Training x-values from a single company
    :param y_train: Training y-values from a single company
    :return: tf.keras.callbacks.History object
    """
    lstm_cp = tf.keras.callbacks.ModelCheckpoint("best_model/", save_best_only=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=tf.keras.metrics.RootMeanSquaredError())
    history = model.fit(x_train, y_train, validation_split=val_ratio, epochs=num_of_epochs, callbacks=[lstm_cp], batch_size=15)

    return history


def fit_model_industry(model, x_train, y_train):
    """
    Trains the neural network model on a single sector.

    :param model: tf.keras.Model object
    :param x_train: Training x-values from a single sector
    :param y_train: Training y-values from a single sector
    :return: tf.keras.callbacks.History object
    """
    lstm_cp = tf.keras.callbacks.ModelCheckpoint("best_model_industry/", save_best_only=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=tf.keras.metrics.RootMeanSquaredError())
    history = model.fit(x_train, y_train, validation_split=val_ratio, epochs=num_of_epochs, callbacks=[lstm_cp])

    return history


def fit_model_vary(model, x_train, y_train, company_symbol):
    """
    Trains the neural network model and saves according to company_symbol.

    :param company_symbol: Company symbol string
    :param model: tf.keras.Model object
    :param x_train: Training x-values from a single sector
    :param y_train: Training y-values from a single sector
    :return: tf.keras.callbacks.History object
    """
    lstm_cp = tf.keras.callbacks.ModelCheckpoint("best_model" + "_" + company_symbol + "/", save_best_only=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=tf.keras.metrics.RootMeanSquaredError())
    history = model.fit(x_train, y_train, validation_split=val_ratio, epochs=num_of_epochs, callbacks=[lstm_cp])

    return history


def get_relative_rmse(predictions, y):
    labels_list = ["Open", "High", "Low", "Close"]
    enumerated_labels = enumerate(labels_list)
    relative_root_mean_squared_error_list = []
    for i, label in enumerated_labels:
        root_mean_squared_error = skm.mean_squared_error(y[:, i], predictions[:, i], squared=False)
        y_mean = np.mean(y[:, i])
        relative_root_mean_squared_error = root_mean_squared_error/y_mean
        print(label + " Relative RMSE:" + str(relative_root_mean_squared_error))

        relative_root_mean_squared_error_list.append(relative_root_mean_squared_error)

    return relative_root_mean_squared_error_list


def train_single_company(dataframe_list_by_symbol):
    scaled_dataframe_list = normalize_by_symbol(dataframe_list_by_symbol)
    train_df, test_df = split_train_test(scaled_dataframe_list)

    train_df_list_by_symbol = group_by_symbol(train_df)
    test_df_list_by_symbol = group_by_symbol(test_df)
    print(train_df_list_by_symbol[COMPANY_INDEX])

    x_train, y_train = convert_to_lstm_input(train_df_list_by_symbol[COMPANY_INDEX])
    input_shape = (np.shape(x_train)[1], np.shape(x_train)[2])
    lstm_model = create_model(input_shape)

    if is_train_model:
        fit_model(lstm_model, x_train, y_train)

    x_test, y_test = convert_to_lstm_input(test_df_list_by_symbol[COMPANY_INDEX])
    best_model = tf.keras.models.load_model("best_model/")

    predictions = best_model.predict(x_test)
    relative_rmse_list = get_relative_rmse(predictions, y_test)
    return relative_rmse_list


def train_single_sector(dataframe_list_by_symbol):
    scaled_dataframe_list = normalize_by_sector(dataframe_list_by_symbol)
    train_df, test_df = split_train_test(scaled_dataframe_list)

    rows_per_company = len(dataframe_list_by_symbol[0])
    num_of_test_rows_per_company = int((1 - train_ratio) * rows_per_company)

    train_df_list_by_sector = group_by_sector(train_df)
    test_df_list_by_sector = group_by_sector(test_df)
    print(train_df_list_by_sector[SECTOR_INDEX])

    x_train, y_train = convert_to_lstm_input(train_df_list_by_sector[SECTOR_INDEX])
    input_shape = (np.shape(x_train)[1], np.shape(x_train)[2])
    lstm_model = create_model(input_shape)

    if is_train_model:
        fit_model_industry(lstm_model, x_train, y_train)

    x_test, y_test = convert_to_lstm_input(test_df_list_by_sector[SECTOR_INDEX])
    best_model = tf.keras.models.load_model("best_model_industry/")

    predictions = best_model.predict(x_test[:num_of_test_rows_per_company])
    relative_rmse_list = get_relative_rmse(predictions, y_test[:num_of_test_rows_per_company])
    return relative_rmse_list


def train_in_sector_by_companies(dataframe_list_by_symbol):
    scaled_by_company_df_list = normalize_by_symbol(dataframe_list_by_symbol)
    company_train_df, company_test_df = split_train_test(scaled_by_company_df_list)

    train_df_list_by_sector_scaled_by_company = group_by_sector(company_train_df)
    test_df_list_by_sector_scaled_by_company = group_by_sector(company_test_df)

    train_df_list_by_company = group_by_symbol(train_df_list_by_sector_scaled_by_company[SECTOR_INDEX])
    test_df_list_by_company = group_by_symbol(test_df_list_by_sector_scaled_by_company[SECTOR_INDEX])

    relative_rmse_list_of_list = []
    for i in range(len(train_df_list_by_company)):
        company_symbol = str(train_df_list_by_company[i].iloc[0, 1])
        print(train_df_list_by_company[i])

        x_train, y_train = convert_to_lstm_input(train_df_list_by_company[i])
        input_shape = (np.shape(x_train)[1], np.shape(x_train)[2])
        lstm_model = create_model(input_shape)

        if is_train_model:
            fit_model_vary(lstm_model, x_train, y_train, company_symbol)

        x_test, y_test = convert_to_lstm_input(test_df_list_by_company[i])
        best_model = tf.keras.models.load_model("best_model" + "_" + company_symbol + "/")

        predictions = best_model.predict(x_test)
        relative_rmse_list = get_relative_rmse(predictions, y_test)
        relative_rmse_list_of_list.append(relative_rmse_list)

    arr = np.array(relative_rmse_list_of_list)
    print(arr)
    labels_list = ["Open Avg RRMSE", "High Avg RRMSE", "Low Avg RRMSE", "Close Avg RRMSE"]
    enumerated_labels = enumerate(labels_list)
    for i, label in enumerated_labels:
        print(label + ": " + str(np.mean(arr[:, i])))


def train_in_sector_by_sector(dataframe_list_by_symbol):
    rows_per_company = len(dataframe_list_by_symbol[0])
    num_of_test_rows_per_company = int((1 - train_ratio) * rows_per_company)

    scaled_by_sector_df_list = normalize_by_sector(dataframe_list_by_symbol)
    sector_train_df, sector_test_df = split_train_test(scaled_by_sector_df_list)

    train_df_list_by_sector = group_by_sector(sector_train_df)
    test_df_list_by_sector = group_by_sector(sector_test_df)

    x_train, y_train = convert_to_lstm_input(train_df_list_by_sector[SECTOR_INDEX])
    input_shape = (np.shape(x_train)[1], np.shape(x_train)[2])
    lstm_model = create_model(input_shape)

    if is_train_model:
        fit_model_industry(lstm_model, x_train, y_train)

    x_test, y_test = convert_to_lstm_input(test_df_list_by_sector[SECTOR_INDEX])
    best_model_industry = tf.keras.models.load_model("best_model_industry/")

    num_of_companies_in_sector = int(len(test_df_list_by_sector[SECTOR_INDEX]) / num_of_test_rows_per_company)
    relative_rmse_list_of_list = []
    for i in range(num_of_companies_in_sector):
        predictions = best_model_industry.predict(x_test[i * num_of_test_rows_per_company:(i + 1) * num_of_test_rows_per_company])
        relative_rmse_list = get_relative_rmse(predictions, y_test[i * num_of_test_rows_per_company:(i + 1) * num_of_test_rows_per_company])
        relative_rmse_list_of_list.append(relative_rmse_list)

    arr = np.array(relative_rmse_list_of_list)
    print(arr)
    labels_list = ["Open Avg RRMSE", "High Avg RRMSE", "Low Avg RRMSE", "Close Avg RRMSE"]
    enumerated_labels = enumerate(labels_list)
    for i, label in enumerated_labels:
        print(label + ": " + str(np.mean(arr[:, i])))


def main():
    filename = r"data.parquet"  # Replace with data.parquet path
    dataframe_list_by_symbol = convert_parquet_to_dataframe_list(filename)
    train_in_sector_by_companies(dataframe_list_by_symbol)


if __name__ == '__main__':
    main()
