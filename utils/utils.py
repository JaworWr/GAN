import os
from datetime import datetime
from typing import Union, Dict

import yaml
from dotmap import DotMap


def plot_sample(ax, sample):
    ax.imshow(sample[0].detach(), cmap="gray", vmin=-1, vmax=1)


def read_config(path: str) -> DotMap:
    with open(path) as f:
        src = f.read()
    d = yaml.safe_load(src)
    d = DotMap(d)
    d._src = src
    return d


def experiment_init(config) -> Dict[str, Union[str, None]]:
    if config.experiment.save:
        experiment_root = config.experiment.get("root", "experiments")
        experiment_name = config.experiment.get("name", datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        checkpoint_path = os.path.join(experiment_root, experiment_name, "model_checkpoints")
        log_path = os.path.join(experiment_root, experiment_name, "logs")
        os.makedirs(checkpoint_path)
        os.makedirs(log_path)
        with open(os.path.join(experiment_root, experiment_name, "config.yml"), "w") as out:
            out.write(config._src)
        return {
            "checkpoint_path": checkpoint_path,
            "log_path": log_path,
        }
    else:
        return {
            "checkpoint_path": None,
            "log_path": None,
        }
