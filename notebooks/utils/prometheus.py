from datetime import timedelta

import numpy as np
import pandas as pd
from prometheus_api_client import PrometheusConnect, Metric


def get_hyperflow_metric_time_serie(parameter: str, hyperflowId: str, params: dict) -> Metric:
    prom = PrometheusConnect(url="http://localhost:9090", disable_ssl=True)

    metric = "hyperflow_job_parameter"
    if parameter.startswith("io_") or parameter.startswith("network_"):
        metric += "_total"

    query = f"sum({metric}{{parameter='{parameter}', hyperflowId='{hyperflowId}'}}) or vector(0)"
    if parameter.startswith("io_") or parameter.startswith("network_"):
        query = f"rate(({query})[60s:])"

    if parameter == "jobs":
        query = f"count(count(hyperflow_job_parameter{{hyperflowId='{hyperflowId}'}}) by (jobId) or count(hyperflow_job_parameter_total{{hyperflowId='{hyperflowId}'}}) by (jobId)) or vector(0)"

    result = prom.custom_query_range(query, **params)[0]
    result["metric"]["__name__"] = ""
    return Metric(result)


def get_time_series(hyperflowId: str, params: dict, parameters: list) -> pd.DataFrame:
    metrics_df = None

    for p in parameters:

        m = get_hyperflow_metric_time_serie(p, hyperflowId, params).metric_values
        m.columns = ["ds", p]

        if metrics_df is None:
            metrics_df = m
        else:
            metrics_df = pd.concat([metrics_df, m[p]], axis=1)
    return metrics_df


def metrics_for_index(hf_data, parameters: list, idx: int, step=5, start_off=300, end_off=5):
    flow = hf_data.loc[idx]
    hyperflowId = flow["hyperflowId"]
    params = {
        "start_time": flow["start"] - timedelta(seconds=start_off),
        "end_time": flow["end"] + timedelta(seconds=end_off),
        "step": f"{step}s",
    }

    return get_time_series(hyperflowId, params, parameters)


def get_series(hf_data, df_static, parameters: list, step=5, n_past=120, n_future=120) -> list:
    return [
        metrics_for_index(hf_data, parameters, idx, step=step, start_off=(n_past - 2 * step), end_off=n_future // 2).select_dtypes(include=np.number)
        for idx in df_static.index
    ]
