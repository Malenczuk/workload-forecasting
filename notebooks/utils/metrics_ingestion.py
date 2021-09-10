import json
from collections import defaultdict
from datetime import datetime
from itertools import chain
from typing import List, Dict

import numpy as np
import pandas as pd
from prometheus_api_client.utils import parse_datetime
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest
from prometheus_client.metrics_core import CounterMetricFamily, GaugeMetricFamily

from metrics_path import *


def parse_metrics_data(providers: List[Provider]) -> pd.DataFrame:
    records = []

    for provider in providers:
        for instance in provider.instances:
            for workflow in instance.workflows:
                records.append({
                    "provider": provider.name,
                    "instance": instance.name,
                    **_parse_job_descriptions(workflow),
                    **_parse_nodes(workflow),
                    **_parse_sys_info(workflow),
                    **_parse_metrics(workflow),
                })

    return pd.DataFrame.from_records(records)


def _parse_job_descriptions(workflow: Workflow) -> dict:
    workflow_data = {
        "jobs": set(),
        "nodes": set()
    }
    with workflow.job_descriptions.open() as file:
        for i, line in enumerate(file):

            data = json.loads(line)

            if i == 0:
                workflow_data["hyperflowId"] = data["hyperflowId"]
                workflow_data["workflowName"] = data["workflowName"]
                workflow_data["size"] = data["size"]

            workflow_data["jobs"].add(data["jobId"])

            if "env" in data:
                workflow_data["nodes"].add(data["env"]["nodeName"])
    return {
        **workflow_data,
        "jobs": len(workflow_data["jobs"]),
        "nodes": len(workflow_data["nodes"])
    }


def _parse_nodes(workflow: Workflow) -> dict:
    if not non_empty_file(workflow.nodes):
        return {}
    workflow_data = {
        "nodes": set()
    }
    with workflow.nodes.open() as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            workflow_data["nodes"].add(line.split()[0])
    return {"nodes": len(workflow_data["nodes"])}


def _parse_sys_info(workflow: Workflow) -> dict:
    workflow_data = {
        "cpu_speed": [],
        "cpu_cores": [],
        "cpu_physical_cores": [],
        "cpu_processors": [],
        "memory": [],
    }
    with workflow.sys_info.open() as file:
        for i, line in enumerate(file):
            data = json.loads(line)

            workflow_data["cpu_speed"].append(float(data["cpu"]["speed"]))
            workflow_data["cpu_cores"].append(int(data["cpu"]["cores"]))
            workflow_data["cpu_physical_cores"].append(int(data["cpu"]["physicalCores"]))
            workflow_data["cpu_processors"].append(int(data["cpu"]["processors"]))
            workflow_data["memory"].append(int(data["mem"]["total"]))
    return {
        "cpu_speed": np.mean(workflow_data["cpu_speed"]),
        "cpu_cores": np.mean(workflow_data["cpu_cores"]),
        "cpu_physical_cores": np.mean(workflow_data["cpu_physical_cores"]),
        "cpu_processors": np.mean(workflow_data["cpu_processors"]),
        "memory": np.mean(workflow_data["memory"])
    }


def _parse_metrics(workflow: Workflow) -> dict:
    def get_timestamp(time: str) -> datetime:
        return parse_datetime(time, settings={'RETURN_AS_TIMEZONE_AWARE': True, 'TIMEZONE': 'UTC'})

    start = None
    end = None

    with workflow.metrics.open() as file:
        for i, line in enumerate(file):
            t = line[10:33]
            if start is None or start > t:
                min_date = t
            if end is None or end < t:
                max_date = t

    return {
        "start": get_timestamp(start),
        "end": get_timestamp(end)
    }


def parse_metrics_labels(providers: List[Provider]) -> pd.DataFrame:
    label_records = {}

    for provider in providers:
        for instance in provider.instances:

            for workflow in instance.workflows:
                job_labels: Dict[str, Dict[str, str]] = defaultdict(dict)

                with workflow.job_descriptions.open() as file:
                    for i, line in enumerate(file):

                        data = json.loads(line)

                        if "name" not in data:
                            continue

                        job_labels[data["jobId"]]["hyperflowId"] = data["hyperflowId"]
                        job_labels[data["jobId"]]["workflowName"] = data["workflowName"]
                        job_labels[data["jobId"]]["size"] = data["size"]
                        job_labels[data["jobId"]]["version"] = data["version"]
                        job_labels[data["jobId"]]["name"] = data["name"]
                        job_labels[data["jobId"]]["executable"] = data["executable"]

                        if "env" in data:
                            job_labels[data["jobId"]]["node"] = data["env"]["nodeName"]
                            job_labels[data["jobId"]]["pod"] = data["env"]["podName"]
                        else:
                            job_labels[data["jobId"]]["node"] = ""
                            job_labels[data["jobId"]]["pod"] = ""

                if non_empty_file(workflow.nodes):
                    with workflow.nodes.open() as file:
                        for i, line in enumerate(file):
                            if i == 0:
                                continue
                            data = line.split()

                            jobId = next(
                                i for i, d in job_labels.items()
                                if f"-{d['name'].lower().replace('_', '-')}-{i.split('-')[-1]}-" in data[1]
                            )

                            job_labels[jobId]["node"] = data[0]
                            job_labels[jobId]["pod"] = data[1]

                label_records.update(job_labels)

    return pd.DataFrame.from_dict(label_records, orient='index', dtype=str)


class Collector(object):

    def __init__(self, file_path: Path, hf_labels, registry=REGISTRY):
        self.file_path: Path = file_path
        self.registry = registry
        self.hf_labels = hf_labels
        self.gauge = None
        self.counter = None
        self.labels = list(hf_labels.columns) + ["jobId", "pid", "parameter", "interface"]
        if registry:
            registry.register(self)

    def collect(self):
        def gen(x):
            yield x

        def get_labels(data: dict) -> list:
            return list(self.hf_labels.loc[data["jobId"]]) + [data["jobId"], str(data["pid"])]

        def get_timestamp(data: dict) -> int:
            return int(parse_datetime(data['time'], settings={'RETURN_AS_TIMEZONE_AWARE': True, 'TIMEZONE': 'UTC'}).timestamp())

        if not self.gauge:

            self.gauge = GaugeMetricFamily("hyperflow_job_parameter", "metrics collected in hyperflow job executor", labels=self.labels)
            self.counter = CounterMetricFamily("hyperflow_job_parameter", "metrics collected in hyperflow job executor", labels=self.labels)
            with self.file_path.open() as file:
                for i, line in enumerate(file):
                    data = json.loads(line)

                    par = data.get("parameter", None)
                    if par and par != 'event':
                        labels = get_labels(data)
                        timestamp = get_timestamp(data)
                        if par == 'network':
                            interface = data['value']['name']
                            for p, v in [(f"network_{n}", v) for n, v in data['value'].items() if n != 'name']:
                                self.counter.add_metric(labels + [p, interface], v, timestamp=timestamp)
                        elif par == 'io':
                            for p, v in [(f"io_{n}", v) for n, v in data['value'].items()]:
                                self.counter.add_metric(labels + [p], v, timestamp=timestamp)
                        else:
                            self.gauge.add_metric(labels + [par], data['value'], timestamp=timestamp)
        return chain(gen(self.gauge), gen(self.counter))

    def unregister(self):

        if self.registry:
            self.registry.unregister(self)


def generate_open_metrics(providers: List[Provider], hf_labels: pd.DataFrame):
    for provider in providers:
        for instance in provider.instances:
            print(f"Provider: {provider.name}, Instance: {instance.name}")

            for workflow in instance.workflows:

                if non_empty_file(workflow.open_metrics):
                    continue

                registry = CollectorRegistry()
                c = Collector(workflow.metrics, hf_labels, registry=registry)

                workflow.open_metrics.write_bytes(generate_latest(registry))
