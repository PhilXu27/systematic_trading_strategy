import pandas as pd


def s25_pre_run(
        labels, features, test_start, test_end, validation_percentage
):
    ##############################
    # Merge Labels with Features #
    ##############################
    bins = labels[["bin"]]
    try:
        bins = bins.map(lambda x: 1.0 if x == 1.0 else 0.0)
    except AttributeError:
        bins = bins.applymap(lambda x: 1.0 if x == 1.0 else 0.0)
    # pandas version conflicts...

    bins.columns = ["label"]
    valid_time_index = bins.index.tolist()
    features = features.loc[valid_time_index]
    data_all = pd.concat([bins, features], axis=1)

    train_data = data_all[data_all.index <= test_start]
    validation_size = int(len(train_data) * validation_percentage)
    val_data = train_data.iloc[-validation_size:]
    hyper_train_data = train_data.iloc[:-validation_size]
    test_data = data_all[(data_all.index > test_start) & (data_all.index <= test_end)]

    X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    X_hyper_train, y_hyper_train = hyper_train_data.iloc[:, 1:], hyper_train_data.iloc[:, 0]
    X_val, y_val = val_data.iloc[:, 1:], val_data.iloc[:, 0]
    X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

    experiment_data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_hyper_train": X_hyper_train,
        "y_hyper_train": y_hyper_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return data_all, experiment_data_dict
