import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.max_rows', None)


is_train_model = True  # False = Use pretrained model
is_by_symbol = False
is_standardize_output = False
is_plot_line = True


COMPANY_INDEX = 117
SECTOR_INDEX = 0

days_backtracked = 1
train_ratio = 0.8
val_ratio = 0.1
num_of_features = 8

initial_label = 0
end_label = 4
num_of_labels = end_label - initial_label

LSTM_units = 64
loss_function = tf.keras.losses.MeanSquaredError()
optimizer_function = tf.keras.optimizers.Adam(learning_rate=0.01)
num_of_epochs = 50


def load_data(filename):
    dataset = pd.read_parquet(filename)
    return dataset


def clean_data(dataframe):
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
            dataframe_list_by_symbol[i]["D-" + str(j) + " Adj Close"] = dataframe_list_by_symbol[i]["Adj Close"].shift(periods=j, axis=0)
            dataframe_list_by_symbol[i]["D-" + str(j) + " Vol"] = dataframe_list_by_symbol[i]["Volume"].shift(periods=j, axis=0)
            dataframe_list_by_symbol[i]["D-" + str(j) + " News Vol Proportion"] = (dataframe_list_by_symbol[i]["News - Volume"] / dataframe_list_by_symbol[i]["News - All News Volume"]).shift(periods=j, axis=0)
            # dataframe_list_by_symbol[i]["D-" + str(j) + "Pos News"] = dataframe_list_by_symbol[i]["News - Positive Sentiment"].shift(periods=j, axis=0)
            # dataframe_list_by_symbol[i]["D-" + str(j) + "Neg News"] = dataframe_list_by_symbol[i]["News - Negative Sentiment"].shift(periods=j, axis=0)
            dataframe_list_by_symbol[i]["D-" + str(j) + " Net News Sentiment"] = (dataframe_list_by_symbol[i]["News - Positive Sentiment"] - dataframe_list_by_symbol[i]["News - Negative Sentiment"]).shift(periods=j, axis=0)
        dataframe_list_by_symbol[i].dropna(inplace=True, ignore_index=True)

    return dataframe_list_by_symbol


# Obtain scaling parameters from training and scale entire data per company
def normalize_by_symbol(dataframe_list_by_symbol):
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


# Normalize x, standardize ALL y (including validation + testing data)
def scale_x_y_by_symbol(dataframe_list_by_symbol):
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

    mean_std_list_per_company = []
    for i in range(num_of_companies):
        mean_std_list = []
        for j in range(3, 3 + num_of_labels):
            col_mean = df_copy_list[i].iloc[:num_of_train_rows, j].mean()
            col_std = df_copy_list[i].iloc[:num_of_train_rows, j].std()
            mean_std_list.append(col_mean)
            mean_std_list.append(col_std)
            if col_std != 0:
                df_copy_list[i].iloc[:, j] -= col_mean
                df_copy_list[i].iloc[:, j] /= col_std

        mean_std_list_per_company.append(mean_std_list)

    return df_copy_list, mean_std_list_per_company


def reverse_standardized_y_by_symbol(y_std, mean_std_list, company_index):
    reversed_y = y_std
    for i, j in zip(range(num_of_labels), range(0, num_of_labels * 2, 2)):
        if mean_std_list[company_index][i + 1] != 0:
            reversed_y[:, i] *= mean_std_list[company_index][j + 1]
            reversed_y[:, i] += mean_std_list[company_index][j]
    return reversed_y


# List of df grouped by symbol -> 1 full df + 1 train-only df -> Group both full and train-only df by sectors
# -> Scaling parameters from train-only df applied to full df -> Full df recombined and grouped by symbol
def normalize_by_sector(dataframe_list_by_symbol):
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


# Normalize x, standardize ALL y (including validation + testing data)
def scale_x_y_by_sector(dataframe_list_by_symbol):
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

    mean_std_list_per_sector = []
    for i in range(num_of_sectors):
        mean_std_list = []
        for j in range(3, 3 + num_of_labels):
            col_mean = train_df_list_by_sector[i].iloc[:, j].mean()
            col_std = train_df_list_by_sector[i].iloc[:, j].std()
            mean_std_list.append(col_mean)
            mean_std_list.append(col_std)
            if col_std != 0:
                df_list_by_sector[i].iloc[:, j] -= col_mean
                df_list_by_sector[i].iloc[:, j] /= col_std
        mean_std_list_per_sector.append(mean_std_list)

    recombined_df = pd.DataFrame()
    for i in range(num_of_sectors):
        recombined_df = pd.concat([recombined_df, df_list_by_sector[i]])

    df_list_by_symbol = group_by_symbol(recombined_df)
    return df_list_by_symbol, mean_std_list_per_sector


def reverse_standardized_y_by_sector(y_std, mean_std_list, sector_index):
    reversed_y = y_std
    for i, j in zip(range(num_of_labels), range(0, num_of_labels * 2, 2)):
        if mean_std_list[sector_index][i + 1] != 0:
            reversed_y[:, i] *= mean_std_list[sector_index][j + 1]
            reversed_y[:, i] += mean_std_list[sector_index][j]
    return reversed_y


def split_train_test(normalized_dataframe_list):
    num_of_companies = len(normalized_dataframe_list)
    num_of_rows = len(normalized_dataframe_list[0])
    num_of_train_rows = int(train_ratio * num_of_rows)

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i in range(num_of_companies):
        train_df = pd.concat([train_df, normalized_dataframe_list[i].iloc[:num_of_train_rows, :]])
        test_df = pd.concat([test_df, normalized_dataframe_list[i].iloc[num_of_train_rows:, :]])

    return train_df, test_df


def group_by_symbol(dataframe):
    grouped_by_symbol = dataframe.groupby(dataframe["Symbol"])
    dataframe_list_by_symbol = []
    for name, data in grouped_by_symbol:
        dataframe_list_by_symbol.append(data)

    return dataframe_list_by_symbol


def group_by_sector(dataframe):
    grouped_by_sector = dataframe.groupby(dataframe["Sector"])
    dataframe_list_by_sector = []
    for name, data in grouped_by_sector:
        dataframe_list_by_sector.append(data)

    return dataframe_list_by_sector


def convert_to_lstm_input(dataset_single):
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
    lstm_model = tf.keras.models.Sequential()

    lstm_model.add(tf.keras.layers.InputLayer(input_shape=shape))

    # lstm_model.add(tf.keras.layers.LSTM(units=LSTM_units, return_sequences=True))
    lstm_model.add(tf.keras.layers.LSTM(units=LSTM_units))
    # lstm_model.add(tf.keras.layers.Dense(units=LSTM_units, kernel_initializer="lecun_normal", activation="selu"))

    lstm_model.add(tf.keras.layers.Dense(units=num_of_labels, activation="linear"))

    return lstm_model


def fit_model(model, x_train, y_train):
    lstm_cp = tf.keras.callbacks.ModelCheckpoint("best_model/", save_best_only=True)

    model.compile(loss=loss_function, optimizer=optimizer_function, metrics=tf.keras.metrics.RootMeanSquaredError())
    history = model.fit(x_train, y_train, validation_split=val_ratio, epochs=num_of_epochs, callbacks=[lstm_cp])

    return history


def fit_model_industry(model, x_train, y_train, rows_per_company):
    lstm_cp = tf.keras.callbacks.ModelCheckpoint("best_model_industry/", save_best_only=True)

    model.compile(loss=loss_function, optimizer=optimizer_function, metrics=tf.keras.metrics.RootMeanSquaredError())

    history = None
    train_rows_per_company = int(train_ratio * rows_per_company)
    num_of_companies = int(len(x_train)/train_rows_per_company)
    for i in range(num_of_companies):
        initial = i * train_rows_per_company
        end = initial + train_rows_per_company
        history = model.fit(x_train[initial:end], y_train[initial:end], validation_split=val_ratio, epochs=num_of_epochs, callbacks=[lstm_cp])

    return history


def compare_predictions(predictions, y):
    if is_plot_line:
        x_axis = np.arange(len(predictions))
        labels_list = ["Open", "High", "Low", "Close"]
        enumerated_labels = enumerate(labels_list)
        for i, label in enumerated_labels:
            plt.plot(x_axis, predictions[:, i], label=label)
            plt.plot(x_axis, y[:, i], label="Actual " + label)
            plt.legend()
            plt.show()


def main():
    filename = r"data.parquet"  # Replace with data.parquet path
    raw_dataframe = load_data(filename)
    cleaned_dataframe = clean_data(raw_dataframe)
    dataframe_list_by_symbol = process_data(cleaned_dataframe)

    mean_std_list = None
    if is_by_symbol:
        if is_standardize_output:
            scaled_dataframe_list, mean_std_list = scale_x_y_by_symbol(dataframe_list_by_symbol)
        # Normalize + Standardize Output Branch
        else:
            scaled_dataframe_list = normalize_by_symbol(dataframe_list_by_symbol)

    # By Sector Branch
    else:
        if is_standardize_output:
            scaled_dataframe_list, mean_std_list = scale_x_y_by_sector(dataframe_list_by_symbol)
        else:
            scaled_dataframe_list = normalize_by_sector(dataframe_list_by_symbol)

    train_df, test_df = split_train_test(scaled_dataframe_list)

    rows_per_company = len(dataframe_list_by_symbol[0])
    num_of_test_rows_per_company = int((1 - train_ratio) * rows_per_company)

    if is_by_symbol:
        train_df_list_by_symbol = group_by_symbol(train_df)
        test_df_list_by_symbol = group_by_symbol(test_df)

        x_train, y_train = convert_to_lstm_input(train_df_list_by_symbol[COMPANY_INDEX])
        input_shape = (np.shape(x_train)[1], np.shape(x_train)[2])
        lstm_model = create_model(input_shape)

        if is_train_model:
            fit_model(lstm_model, x_train, y_train)

        x_test, y_test = convert_to_lstm_input(test_df_list_by_symbol[COMPANY_INDEX])
        best_model = tf.keras.models.load_model("best_model/")

        if is_standardize_output:
            standardized_predictions = best_model.predict(x_test)
            predictions = reverse_standardized_y_by_symbol(standardized_predictions, mean_std_list, COMPANY_INDEX)
            y_test = reverse_standardized_y_by_symbol(y_test, mean_std_list, COMPANY_INDEX)
        else:
            predictions = best_model.predict(x_test)

        compare_predictions(predictions, y_test)

    # By Sector Branch
    else:
        train_df_list_by_sector = group_by_sector(train_df)
        test_df_list_by_sector = group_by_sector(test_df)

        x_train, y_train = convert_to_lstm_input(train_df_list_by_sector[SECTOR_INDEX])
        input_shape = (np.shape(x_train)[1], np.shape(x_train)[2])
        lstm_model = create_model(input_shape)

        if is_train_model:
            fit_model_industry(lstm_model, x_train, y_train, rows_per_company)

        x_test, y_test = convert_to_lstm_input(test_df_list_by_sector[SECTOR_INDEX])
        best_model = tf.keras.models.load_model("best_model_industry/")

        if is_standardize_output:
            standardized_predictions = best_model.predict(x_test[:num_of_test_rows_per_company])
            predictions = reverse_standardized_y_by_sector(standardized_predictions, mean_std_list, SECTOR_INDEX)
            y_test = reverse_standardized_y_by_sector(y_test, mean_std_list, SECTOR_INDEX)
        else:
            predictions = best_model.predict(x_test[:num_of_test_rows_per_company])

        compare_predictions(predictions, y_test[:num_of_test_rows_per_company])


if __name__ == '__main__':
    main()
