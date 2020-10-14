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
