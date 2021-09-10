from dataclasses import dataclass
from pathlib import Path
from typing import List

# Input files gathered from workflow execution
metrics_jsonl = "metrics.jsonl"
job_descriptions_jsonl = "job_descriptions.jsonl"
sys_info_jsonl = "sys_info.jsonl"
nodes_log = "nodes.log"
# Output file name for metrics in OpenMetric format
metrics_om = "metrics.om"


@dataclass(frozen=True)
class Workflow:
    """
    Helper class representing dictionary containing metrics for a given workflow execution
    """
    path: Path

    @property
    def job_descriptions(self) -> Path:
        return self.path.joinpath(job_descriptions_jsonl)

    @property
    def sys_info(self) -> Path:
        return self.path.joinpath(sys_info_jsonl)

    @property
    def metrics(self) -> Path:
        return self.path.joinpath(metrics_jsonl)

    @property
    def open_metrics(self) -> Path:
        return self.path.joinpath(metrics_om)

    @property
    def nodes(self) -> Path:
        return self.path.joinpath(nodes_log)


@dataclass(frozen=True)
class Instance:
    """
    Helper class representing dictionary containing workflow execution metrics for a given machine instance
    """
    name: str
    path: Path
    workflows: List[Workflow]


@dataclass(frozen=True)
class Provider:
    """
    Helper class representing dictionary containing machine instances for a given cloud provider
    """
    name: str
    path: Path
    instances: List[Instance]


def logs(base_path: Path) -> List[Provider]:
    """
    If a directory containing workflow metrics has a file name ".skip" it will be removed from returned list
    :param base_path: path of base directory containing metrics from Hyperflow executions
    :return: list of Providers
    """
    return [
        Provider(
            name=provider_path.name,
            path=provider_path,
            instances=[
                Instance(
                    name=instance_path.name,
                    path=instance_path,
                    workflows=[
                        Workflow(path=workflow_path)
                        for workflow_path in instance_path.iterdir()
                        if workflow_path.is_dir() and not workflow_path.joinpath(".skip").is_file()
                    ]
                )
                for instance_path in provider_path.iterdir()
                if instance_path.is_dir()
            ]
        )
        for provider_path in base_path.iterdir()
        if provider_path.is_dir()
    ]


def non_empty_file(path: Path) -> bool:
    """
    :param path: path of a file
    :return: True if given path is a file and is not empty otherwise return False
    """
    return path.is_file() and path.stat().st_size > 0
