from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd

parameters = [
    "jobs",
    "cpu",
    "memory",
    "io_read",
    "io_readReal",
    "io_readSyscalls",
    "io_write",
    # "io_writeCancelled",
    "io_writeReal",
    "io_writeSyscalls",
    "network_rxBytes",
    # "network_rxCompressed",
    # "network_rxDrop",
    # "network_rxErrors",
    # "network_rxFifo",
    # "network_rxFrame",
    # "network_rxMulticast",
    "network_rxPackets",
    "network_txBytes",
    # "network_txCarrier",
    # "network_txColls",
    # "network_txCompressed",
    # "network_txDrop",
    # "network_txErrors",
    # "network_txFifo",
    "network_txPackets",
]


def get_hf_data(fname="../data/hf-data.pkl.gzip") -> pd.DataFrame:
    return pd.read_pickle(fname, compression="gzip")


def get_hf_max_values(fname="../data/hf-max_values.pkl.gzip") -> pd.DataFrame:
    return pd.read_pickle(fname, compression="gzip")


def get_hf_static_data(df: pd.DataFrame) -> pd.DataFrame:
    col = ['size', 'jobs', 'nodes', 'cpu_speed', 'cpu_cores', 'cpu_physical_cores', 'cpu_processors', 'memory', 'workflowName']
    hf_static = pd.get_dummies(df[col], columns=["workflowName"])
    hf_static["size"] = pd.to_numeric(hf_static["size"], downcast="float")
    return hf_static


def get_max_value_in_series(df: pd.DataFrame, get_metrics: Callable[[str, dict], pd.DataFrame]) -> pd.DataFrame:
    rows = {}

    for index, flow in df.iterrows():

        hyperflowId = flow["hyperflowId"]
        params = {
            "start_time": flow["start"] - timedelta(seconds=5),
            "end_time": flow["end"] + timedelta(seconds=5),
            "step": "5s",
        }
        try:

            metrics_df = get_metrics(hyperflowId, params)
            rows[index] = metrics_df.max()[1:]

        except Exception as e:
            print(f"Getting time series failed for {index} - {flow['workflowName']} {flow['size']}")

    return pd.DataFrame.from_dict(rows, orient='index')


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    validate = df.loc[perm[train_end:validate_end]]
    test = df.loc[perm[validate_end:]]
    return train, validate, test


def split_many_2_many(series, n_past, n_future, split_stride=1):
    X, y = list(), list()

    for window_start in range(0, len(series), split_stride):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)

    return np.array(X), np.array(y)


def dataset_split_many_2_many(static, series, n_past, n_future, features, features_pred, split_stride=1):
    X = []
    y = []
    for i in range(static.shape[0]):
        X_series, y_series = split_many_2_many(series[i], n_past, n_future, split_stride)
        X_static = np.repeat([static[i]], X_series.shape[0], 0)
        if y_series.size > 0:
            X.append((X_series[:, :, features], X_static))
            y.append(y_series[:, :, features_pred])
    return X, y
