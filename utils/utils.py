def plot_sample(ax, sample):
    ax.imshow(sample[0].detach(), cmap="gray", vmin=-1, vmax=1)
